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
    # Handle common prefixes
    prefixes = ['anthropic/', 'google/', 'openai/']
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

def get_model_names(df):
    """Extract model names from columns that end with '_score'."""
    return [normalize_model_name(col[:-6]) for col in df.columns if col.endswith('_score')]

def load_model_data(model_files):
    """Load and combine data for each model, handling multiple files if present."""
    model_data = {}
    observations_per_model = {}
    for model, files in model_files.items():
        # Track total observations for this model
        total_obs = 0
        
        if len(files) == 1:
            # Single file case - load directly
            df = load_and_clean_data(files[0])
        else:
            # Multiple files case - combine data with resampling
            dfs = [load_and_clean_data(f) for f in files]
            
            # Keep non-score columns from first file
            df = dfs[0].copy()
            score_cols = [c for c in df.columns if c.endswith('_score')]
            
            # Average only the score columns across files
            for col in score_cols:
                scores = [df[col].fillna(0) for df in dfs]  # Handle NaN values
                df[col] = pd.concat(scores, axis=1).mean(axis=1)
        
        # Count total observations
        total_obs = len(df) * len(files)
        
        # Normalize column names by removing company prefixes
        score_cols = [c for c in df.columns if c.endswith('_score')]
        rename_map = {col: normalize_model_name(col) for col in score_cols}
        df = df.rename(columns=rename_map)
        
        model_data[model] = df
        observations_per_model[model] = total_obs
    
    return model_data, observations_per_model

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
                scores = judge_df[col].fillna(0)  # Handle NaN values
                if len(scores) > 0:
                    S[i,j] = scores.mean()
    
    # Calculate raw scores (R_j)
    raw_scores = np.zeros(n)
    for i, model in enumerate(models):
        df = model_data[model]
        score_cols = [col for col in df.columns if col.endswith('_score')]
        if score_cols:
            scores = df[score_cols].fillna(0).values  # Handle NaN values
            if scores.size > 0:
                raw_scores[i] = scores.mean()
    
    # Calculate judge harshness (H_j)
    judge_harshness = np.zeros(n)
    for j, judge_model in enumerate(models):
        other_scores = []
        score_col = f"{judge_model}_score"
        
        for i, (model, df) in enumerate(model_data.items()):
            if i != j and score_col in df.columns:
                scores = df[score_col].fillna(0).tolist()  # Handle NaN values
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

def compute_comparison(data_A, data_B, n_files_A, n_files_B, tags=None, paired=False, clustered=False):
    """Calculate difference, standard error, 95% CI, p-value."""
    d_bar = np.mean(data_A) - np.mean(data_B)
    
    if paired:
        diffs = data_A - data_B
        base_var = np.var(diffs, ddof=1) / len(diffs)
    else:
        base_var = (np.var(data_A, ddof=1)/len(data_A) + 
                   np.var(data_B, ddof=1)/len(data_B))
    
    if clustered:
        clusters = pd.Series(tags).fillna('uncategorized')
        n_clusters = clusters.nunique()
        
        if paired:
            df_diff = pd.DataFrame({'diff': diffs, 'cluster': clusters})
            cluster_means = df_diff.groupby('cluster')['diff'].agg(['mean','count'])
            between_var = np.var(cluster_means['mean'], ddof=1)
            se = np.sqrt(between_var / len(cluster_means) + base_var)
        else:
            df_data = pd.DataFrame({'A': data_A, 'B': data_B, 'cluster': clusters})
            cluster_means = df_data.groupby('cluster')[['A','B']].mean()
            cluster_diffs = cluster_means['A'] - cluster_means['B']
            between_var = np.var(cluster_diffs, ddof=1)
            se = np.sqrt(between_var / len(cluster_means) + base_var)
    else:
        se = np.sqrt(base_var)
        n_clusters = np.nan
    
    ci_low = d_bar - 1.96*se
    ci_high = d_bar + 1.96*se
    p_val = 0.0 if se == 0 else 2*(1 - norm.cdf(abs(d_bar / se)))
    
    return {
        'Mean Difference': d_bar,
        'SE': se,
        '95% CI': (ci_low, ci_high),
        'p-value': p_val,
        'N': len(data_A) * n_files_A,  # Total observations including all files
        '#Clusters': n_clusters
    }

def analyze_evaluation_results(base_path, models):
    """
    Primary analysis function that:
      - Loads all result files for specified models
      - Computes analytical biases
      - Produces comparison tables
      - Handles multiple runs per model if available
    """
    # Get all relevant files for each model
    model_files = get_model_files(base_path, models)
    
    # Load and combine data for each model
    model_data, observations_per_model = load_model_data(model_files)
    
    # Compute analytical biases
    bias_results = compute_analytical_biases(model_data, models)
    
    def get_adjusted_data():
        adjusted_data = {}
        for model, df in model_data.items():
            new_df = df.copy()
            for col in [c for c in df.columns if c.endswith('_score')]:
                j_idx = models.index(col[:-6])
                # Subtract judge harshness
                new_df[col] -= bias_results['judge_harshness'][models[j_idx]]
                # If this is self-assessment, subtract self preference
                if j_idx == models.index(model):
                    new_df[col] -= bias_results['self_preference'][models[j_idx]]
            adjusted_data[model] = new_df
        return adjusted_data
    
    def get_arrays(use_adjusted=False):
        """Return (score_arrays, tags_arrays)."""
        data_source = get_adjusted_data() if use_adjusted else model_data
        score_arrays = [df.filter(regex='_score$').mean(axis=1).values 
                       for df in data_source.values()]
        tags_arrays = [df['tag3'].values if 'tag3' in df.columns 
                      else np.array(['NA']*len(df)) for df in data_source.values()]
        return score_arrays, tags_arrays
    
    def format_comparisons(model_pairs, paired, clustered, adjusted_label):
        score_arrays, tags_arrays = get_arrays(use_adjusted=(adjusted_label=='Debiased'))
        results = []
        for i, j in model_pairs:
            dA = score_arrays[i]
            dB = score_arrays[j]
            tA = tags_arrays[i] if clustered else None
            
            # Get number of files for each model
            n_files_A = len(model_files[models[i]])
            n_files_B = len(model_files[models[j]])
            
            stats = compute_comparison(dA, dB, n_files_A, n_files_B, 
                                    tags=tA, paired=paired, clustered=clustered)
            
            row = {
                'Model 1': models[i],
                'Model 2': models[j],
            }
            if adjusted_label == 'Debiased':
                row['Mean M1'] = bias_results['adjusted_scores'][models[i]]
                row['Mean M2'] = bias_results['adjusted_scores'][models[j]]
            else:
                row['Mean M1'] = np.mean(dA)
                row['Mean M2'] = np.mean(dB)
            row.update(stats)
            results.append(row)
        return pd.DataFrame(results)
    
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
        df_raw = format_comparisons(model_pairs, paired=p, clustered=c, 
                                  adjusted_label='Raw')
        all_comparisons.append((f"{desc} (Raw)", df_raw))
    for desc, p, c in comparison_specs:
        df_deb = format_comparisons(model_pairs, paired=p, clustered=c, 
                                  adjusted_label='Debiased')
        all_comparisons.append((f"{desc} (Debiased)", df_deb))
    
    # Print comparison tables
    print("\n=== Comparison Tables ===")
    for title, df in all_comparisons:
        if df.empty:
            continue
        df_print = df.copy()
        for col in ['Mean M1','Mean M2','Mean Difference','SE','p-value']:
            if col in df_print.columns:
                df_print[col] = df_print[col].map('{:.4f}'.format)
        if '95% CI' in df_print.columns:
            df_print['95% CI'] = df_print['95% CI'].apply(
                lambda x: f"({x[0]:.4f}, {x[1]:.4f})")
        if 'N' in df_print.columns:
            df_print['N'] = df_print['N'].astype(int)
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
