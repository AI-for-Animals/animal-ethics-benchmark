import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import glob

def load_and_clean_data(file_path):
    """Load CSV and convert all *_score columns to numeric."""
    df = pd.read_csv(file_path)
    score_cols = [c for c in df.columns if c.endswith('_score')]
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
    return df

def get_model_files(base_path, models):
    """Get all result files for each model from the base path."""
    model_files = {}
    for model in models:
        pattern = os.path.join(base_path, f"results_{model}*.csv")
        files = glob.glob(pattern)
        if files:  # Only include model if files exist
            model_files[model] = files
    return model_files

def normalize_model_name(name):
    """Remove company prefix from model name if present."""
    prefixes = ['anthropic/', 'google/', 'openai/']
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

def get_model_names(df):
    """Extract model names from columns that end with '_score'."""
    return [normalize_model_name(col[:-6]) for col in df.columns if col.endswith('_score')]

def load_model_data(model_files):
    """
    Load and combine data for each model, handling multiple files if present.
    Now APPENDS files instead of averaging for correct SE calculations.
    """
    model_data = {}
    model_meta = {}
    
    for model, files in model_files.items():
        if len(files) == 0:
            continue
        elif len(files) == 1:
            df = load_and_clean_data(files[0])
            total_obs = len(df)
            sample_count = 1
            rows_per_sample = total_obs
        else:
            # Multiple files case - append the dataframes
            dfs = [load_and_clean_data(f) for f in files]
            df = pd.concat(dfs, axis=0, ignore_index=True)
            total_obs = len(df)
            sample_count = len(files)
            rows_per_sample = total_obs // sample_count

        # Normalize column names
        score_cols = [c for c in df.columns if c.endswith('_score')]
        rename_map = {col: normalize_model_name(col) for col in score_cols}
        df = df.rename(columns=rename_map)

        model_data[model] = df
        model_meta[model] = {
            'sample_count': sample_count,
            'rows_per_sample': rows_per_sample,
            'total_obs': total_obs
        }
    
    return model_data, model_meta

def compute_analytical_biases(model_data, models):
    """Calculate biases using analytical formulas."""
    n = len(models)
    
    # Create score matrix S[i,j] where i=judge, j=model being judged
    S = np.zeros((n, n))
    for i, judge_model in enumerate(models):
        judge_df = model_data[judge_model]
        for j, target_model in enumerate(models):
            col = f"{target_model}_score"
            if col in judge_df.columns:
                scores = judge_df[col].fillna(0)
                if len(scores) > 0:
                    S[i,j] = scores.mean()
    
    # Calculate raw scores (R_j)
    raw_scores = np.zeros(n)
    for i, model in enumerate(models):
        df = model_data[model]
        score_cols = [col for col in df.columns if col.endswith('_score')]
        if score_cols:
            scores = df[score_cols].fillna(0).values
            if scores.size > 0:
                raw_scores[i] = scores.mean()
    
    # Calculate judge harshness (H_j)
    judge_harshness = np.zeros(n)
    for j, judge_model in enumerate(models):
        other_scores = []
        score_col = f"{judge_model}_score"
        for i, (mod, df) in enumerate(model_data.items()):
            if i != j and score_col in df.columns:
                scores = df[score_col].fillna(0).tolist()
                if scores:
                    other_scores.extend(scores)
        if other_scores:
            judge_harshness[j] = np.mean(other_scores)
    
    # Calculate self preference (P_i)
    self_preference = np.zeros(n)
    for i in range(n):
        self_assessment = S[i,i]
        judge_harshness_i = judge_harshness[i]
        if not np.isnan(self_assessment) and not np.isnan(judge_harshness_i):
            self_preference[i] = self_assessment - judge_harshness_i
    
    # Calculate adjusted scores (A_j)
    adjusted_scores = np.zeros(n)
    for j, model in enumerate(models):
        scores_from_others = []
        for i, judge_model in enumerate(models):
            if i != j:
                score_col = f"{judge_model}_score"
                if score_col in model_data[model].columns:
                    scores = model_data[model][score_col].fillna(0)
                    if len(scores) > 0:
                        scores_from_others.append(scores.mean())
        if scores_from_others:
            sum_others = np.sum(scores_from_others)
            if not np.isnan(judge_harshness[j]):
                adjusted_scores[j] = (sum_others + judge_harshness[j]) / n
    
    # Calculate total bias
    total_bias = raw_scores - adjusted_scores
    
    results = {
        'raw_scores': dict(zip(models, raw_scores)),
        'adjusted_scores': dict(zip(models, adjusted_scores)),
        'judge_harshness': dict(zip(models, judge_harshness)),
        'self_preference': dict(zip(models, self_preference)),
        'total_bias': dict(zip(models, total_bias))
    }
    
    # Print results
    print("\n=== Analytical Bias Analysis ===")
    for metric, values in results.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        for model, score in values.items():
            print(f"{model:>40} {score:.4f}")
    
    return results

def compute_comparison(data_A, data_B, paired=False, clustered=False, tags=None):
    """
    Calculate difference, standard error, 95% CI, p-value, plus the final N used.
    N is now:
    - len(data_A) if paired (should equal len(data_B))
    - len(data_A) + len(data_B) if unpaired
    """
    if paired:
        # Must have same length arrays
        diffs = data_A - data_B
        d_bar = np.mean(diffs)
        base_var = np.var(diffs, ddof=1) / len(diffs) if len(diffs) > 1 else 0.0
        effective_n = len(diffs)
    else:
        d_bar = np.mean(data_A) - np.mean(data_B)
        varA = np.var(data_A, ddof=1)/len(data_A) if len(data_A) > 1 else 0.0
        varB = np.var(data_B, ddof=1)/len(data_B) if len(data_B) > 1 else 0.0
        base_var = varA + varB
        effective_n = len(data_A) + len(data_B)
    
    # Clustered adjustment (optional)
    if clustered:
        clusters = pd.Series(tags).fillna('uncategorized')
        n_clusters = clusters.nunique()
        
        if paired:
            df_diff = pd.DataFrame({'diff': diffs, 'cluster': clusters})
            cluster_means = df_diff.groupby('cluster')['diff'].agg(['mean','count'])
            between_var = np.var(cluster_means['mean'], ddof=1) if len(cluster_means) > 1 else 0.0
            se = np.sqrt(between_var / len(cluster_means) + base_var)
        else:
            df_data = pd.DataFrame({'A': data_A, 'B': data_B, 'cluster': clusters})
            cluster_means = df_data.groupby('cluster')[['A','B']].mean()
            cluster_diffs = cluster_means['A'] - cluster_means['B']
            between_var = np.var(cluster_diffs, ddof=1) if len(cluster_diffs) > 1 else 0.0
            se = np.sqrt(between_var / len(cluster_means) + base_var)
    else:
        n_clusters = np.nan
        se = np.sqrt(base_var)

    ci_low = d_bar - 1.96 * se
    ci_high = d_bar + 1.96 * se
    p_val = 0.0 if se == 0 else 2.0 * (1.0 - norm.cdf(abs(d_bar / se)))

    return {
        'Mean Difference': d_bar,
        'SE': se,
        '95% CI': (ci_low, ci_high),
        'p-value': p_val,
        'N': effective_n,
        '#Clusters': n_clusters
    }

def analyze_evaluation_results(base_path, models):
    """
    Primary analysis function that:
    - Loads all result files for specified models (appending multiple files)
    - Computes analytical biases
    - Produces comparison tables
    - Shows # samples, rows_per_sample, and total N for each comparison
    """
    model_files = get_model_files(base_path, models)
    model_data, model_meta = load_model_data(model_files)
    bias_results = compute_analytical_biases(model_data, models)
    
    def get_adjusted_data():
        adjusted_data = {}
        for model, df in model_data.items():
            new_df = df.copy()
            for col in [c for c in df.columns if c.endswith('_score')]:
                j_idx = models.index(col[:-6])
                # Subtract judge harshness
                new_df[col] -= bias_results['judge_harshness'][models[j_idx]]
                # If self-assessment, subtract self preference
                if j_idx == models.index(model):
                    new_df[col] -= bias_results['self_preference'][models[j_idx]]
            adjusted_data[model] = new_df
        return adjusted_data
    
    def get_arrays(use_adjusted=False):
        data_source = get_adjusted_data() if use_adjusted else model_data
        score_arrays = []
        tags_arrays = []
        for model in models:
            df = data_source[model]
            arr = df.filter(regex='_score$').mean(axis=1).values
            score_arrays.append(arr)
            if 'tag3' in df.columns:
                tags_arrays.append(df['tag3'].values)
            else:
                tags_arrays.append(np.array(['NA']*len(df)))
        return score_arrays, tags_arrays
    
    def format_comparisons(model_pairs, paired, clustered, adjusted_label):
        score_arrays, tags_arrays = get_arrays(use_adjusted=(adjusted_label=='Debiased'))
        rows = []
        for i, j in model_pairs:
            dataA = score_arrays[i]
            dataB = score_arrays[j]
            tagsA = tags_arrays[i] if clustered else None
            
            # Get metadata for both models
            metaA = model_meta[models[i]]
            metaB = model_meta[models[j]]
            
            stats = compute_comparison(
                data_A=dataA,
                data_B=dataB,
                paired=paired,
                clustered=clustered,
                tags=tagsA
            )
            
            row = {
                'Model 1': models[i],
                'Model 2': models[j],
                '#Samples_A': metaA['sample_count'],
                'RowsPerSample_A': metaA['rows_per_sample'],
                'N_A': metaA['total_obs'],
                '#Samples_B': metaB['sample_count'],
                'RowsPerSample_B': metaB['rows_per_sample'],
                'N_B': metaB['total_obs'],
            }
            
            if adjusted_label == 'Debiased':
                row['Mean M1'] = bias_results['adjusted_scores'][models[i]]
                row['Mean M2'] = bias_results['adjusted_scores'][models[j]]
            else:
                row['Mean M1'] = np.mean(dataA)
                row['Mean M2'] = np.mean(dataB)
            
            row.update(stats)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    # Generate all comparison tables
    comparison_specs = [
        ('Neither Pairwise nor Clustered', False, False),
        ('Pairwise Unclustered', True, False),
        ('Clustered Unpaired', False, True),
        ('Pairwise-and-Clustered', True, True),
    ]
    
    model_pairs = [(i, j) for i in range(len(models)) 
                   for j in range(i+1, len(models))]
    
    all_comparisons = []
    for desc, p, c in comparison_specs:
        df_raw = format_comparisons(model_pairs, paired=p, clustered=c, adjusted_label='Raw')
        all_comparisons.append((f"{desc} (Raw)", df_raw))
    for desc, p, c in comparison_specs:
        df_deb = format_comparisons(model_pairs, paired=p, clustered=c, adjusted_label='Debiased')
        all_comparisons.append((f"{desc} (Debiased)", df_deb))
    
    # Print comparison tables
    print("\n=== Comparison Tables ===")
    for title, df in all_comparisons:
        if df.empty:
            continue
        df_print = df.copy()
        
        # Format numeric columns
        float_cols = ['Mean M1', 'Mean M2', 'Mean Difference', 'SE', 'p-value']
        for col in float_cols:
            if col in df_print.columns:
                df_print[col] = df_print[col].map('{:.4f}'.format)
        
        if '95% CI' in df_print.columns:
            df_print['95% CI'] = df_print['95% CI'].apply(
                lambda x: f"({x[0]:.4f}, {x[1]:.4f})")
        
        # Format integer columns
        int_cols = ['N', '#Clusters', '#Samples_A', 'RowsPerSample_A', 'N_A',
                   '#Samples_B', 'RowsPerSample_B', 'N_B']
        for col in int_cols:
            if col in df_print.columns:
                df_print[col] = df_print[col].astype('Int64')  # handles NaN better than int
        
        print(f"\n--- {title} ---")
        print(df_print.to_string(index=False))
    
    return {'comparisons': dict(all_comparisons)}

if __name__ == "__main__":
    # Example usage with multiple models and base path
    BASE_PATH = "/content/drive/MyDrive/eval_outputs"
    MODELS = [
        "claude-3-5-sonnet-20241022",
        "gemini-1.5-pro-002",
        "gpt-4o-2024-08-06"
    ]
    
    analyze_evaluation_results(BASE_PATH, MODELS)
