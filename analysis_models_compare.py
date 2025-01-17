import numpy as np
import pandas as pd
from scipy.stats import norm


def load_and_clean_data(file_path):
    """Load CSV and convert all *_score columns to numeric."""
    df = pd.read_csv(file_path)
    score_cols = [c for c in df.columns if c.endswith('_score')]
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
    return df

def get_model_names(df):
    """Extract model names from columns that end with '_score'."""
    return [col[:-6] for col in df.columns if col.endswith('_score')]

def compute_analytical_biases(dfs, models):
    """
    Calculate biases using analytical formulas.
    
    Args:
        dfs: List of dataframes, each containing scores from one judge
        models: List of model names
        
    Returns:
        Dictionary containing:
        - raw_scores: Average scores received by each model
        - adjusted_scores: Debiased scores
        - judge_harshness: Average rating given to other models
        - self_preference: How much better a model rates itself vs others
        - total_bias: Difference between raw and adjusted scores
    """
    n = len(models)
    
    # Create score matrix S[i,j] where i=judge, j=model being judged
    S = np.zeros((n, n))
    for i, df in enumerate(dfs):
        for j, model in enumerate(models):
            col = f"{model}_score"
            if col in df.columns:
                S[i,j] = df[col].mean()
    
    # Calculate raw scores (R_j) - mean of ALL scores in each model's results file
    raw_scores = np.zeros(n)
    for i, df in enumerate(dfs):
        # Get all score columns
        score_cols = [col for col in df.columns if col.endswith('_score')]
        # Take mean of ALL scores in this file (all assessments of this model)
        raw_scores[i] = df[score_cols].values.mean()
    
    # Calculate judge harshness (H_j)
    judge_harshness = np.zeros(n)
    for j, judge_model in enumerate(models):
        # Get scores this judge gave to OTHER models
        other_scores = []
        score_col = f"{judge_model}_score"  # column name for this judge's scores
        
        # Look for this judge's scores in other models' files
        for i, df in enumerate(dfs):
            if i != j and score_col in df.columns:  # exclude self-assessment
                other_scores.extend(df[score_col].tolist())
                
        # Calculate average score this judge gave to others
        judge_harshness[j] = np.mean(other_scores)
    
    # Calculate self preference (P_i) = S_ii - H_i 
    # (how much better a model rates itself compared to how it rates others)
    self_preference = np.zeros(n)
    for i in range(n):
        self_assessment = S[i,i]  # S_ii
        judge_harshness_i = judge_harshness[i]  # H_i (already calculated above)
        self_preference[i] = self_assessment - judge_harshness_i
    
    # Calculate adjusted scores (A_j)
    adjusted_scores = np.zeros(n)
    for j, model in enumerate(models):
        # Get AVERAGE scores received from OTHER judges
        scores_from_others = []
        for i, judge_model in enumerate(models):
            if i != j:  # exclude self-assessment
                score_col = f"{judge_model}_score"
                if score_col in dfs[j].columns:
                    # Take mean of this judge's column first
                    scores_from_others.append(dfs[j][score_col].mean())
        
        # Sum of average scores from other judges
        sum_others = np.sum(scores_from_others)
        adjusted_scores[j] = (sum_others + judge_harshness[j]) / n
    
    # Calculate total bias (should equal P_j/n)
    total_bias = raw_scores - adjusted_scores
    
    # Create results dictionary
    results = {
        'raw_scores': dict(zip(models, raw_scores)),
        'adjusted_scores': dict(zip(models, adjusted_scores)),
        'judge_harshness': dict(zip(models, judge_harshness)),
        'self_preference': dict(zip(models, self_preference)),
        'total_bias': dict(zip(models, total_bias))
    }
    
    # Print detailed results
    print("\n=== Analytical Bias Analysis ===")
    print("\nRaw Scores (R_j):")
    for model, score in results['raw_scores'].items():
        print(f"{model:>40} {score:.4f}")
        
    print("\nAdjusted Scores (A_j):")
    for model, score in results['adjusted_scores'].items():
        print(f"{model:>40} {score:.4f}")
        
    print("\nJudge Harshness (H_j):")
    for model, score in results['judge_harshness'].items():
        print(f"{model:>40} {score:.4f}")
        
    print("\nSelf Preference (P_j = S_ii - H_i):")
    for model, score in results['self_preference'].items():
        print(f"{model:>40} {score:.4f}")
        
    print("\nTotal Bias (R_j - A_j):")
    for model, score in results['total_bias'].items():
        print(f"{model:>40} {score:.4f}")
    
    # Print detailed calculation for each model
    print("\n=== Detailed Self-Preference Calculations ===")
    for i, model in enumerate(models):
        print(f"\n{model}:")
        print(f"  Self-assessment (S_ii):     {S[i,i]:.4f}")
        print(f"  Judge harshness (H_i):      {judge_harshness[i]:.4f}")
        print(f"  Self-preference (S_ii - H_i): {self_preference[i]:.4f}")
        
    return results

def compute_comparison(data_A, data_B, tags=None, paired=False, clustered=False):
    """
    Calculate difference, standard error, 95% CI, p-value.
    Also return the number of data points (N) and #Clusters if clustered.
    """
    d_bar = np.mean(data_A) - np.mean(data_B)
    if paired:
        diffs = data_A - data_B
        base_var = np.var(diffs, ddof=1) / len(diffs)
    else:
        base_var = (np.var(data_A, ddof=1)/len(data_A)
                    + np.var(data_B, ddof=1)/len(data_B))

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
        'N': len(data_A),
        '#Clusters': n_clusters
    }

def analyze_evaluation_results(dfs, file_paths):
    """
    Primary analysis function that:
      - Computes analytical biases
      - Produces comparison tables
      - Prints cluster counts for tag3
    """
    models = get_model_names(dfs[0])
    bias_results = compute_analytical_biases(dfs, models)
    
    # Build "adjusted" data by subtracting judge_bias & self_bias
    def get_adjusted_data():
        adjusted_dfs = []
        for i, df in enumerate(dfs):
            new_df = df.copy()
            for col in [c for c in df.columns if c.endswith('_score')]:
                j_idx = models.index(col[:-6])
                # Subtract judge harshness
                new_df[col] -= bias_results['judge_harshness'][models[j_idx]]
                # If this is self-assessment, subtract self preference
                if j_idx == i:
                    new_df[col] -= bias_results['self_preference'][models[j_idx]]
            adjusted_dfs.append(new_df)
        return adjusted_dfs

    def get_arrays(use_adjusted=False):
        """
        Return (score_arrays, tags_arrays).
        """
        data_source = get_adjusted_data() if use_adjusted else dfs
        score_arrays = [df.filter(regex='_score$').mean(axis=1).values for df in data_source]
        tags_arrays = [df['tag3'].values if 'tag3' in df.columns else np.array(['NA']*len(df))
                       for df in data_source]
        return score_arrays, tags_arrays

    def format_comparisons(model_pairs, paired, clustered, adjusted_label):
        score_arrays, tags_arrays = get_arrays(use_adjusted=(adjusted_label=='Debiased'))
        results = []
        for i, j in model_pairs:
            dA = score_arrays[i]
            dB = score_arrays[j]
            tA = tags_arrays[i] if clustered else None

            stats = compute_comparison(dA, dB, tags=tA, paired=paired, clustered=clustered)
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
    model_pairs = [(i, j) for i in range(len(models)) for j in range(i+1, len(models))]

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
        for col in ['Mean M1','Mean M2','Mean Difference','SE','p-value']:
            if col in df_print.columns:
                df_print[col] = df_print[col].map('{:.4f}'.format)
        if '95% CI' in df_print.columns:
            df_print['95% CI'] = df_print['95% CI'].apply(lambda x: f"({x[0]:.4f}, {x[1]:.4f})")
        if 'N' in df_print.columns:
            df_print['N'] = df_print['N'].astype(int)
        print(f"\n--- {title} ---")
        print(df_print.to_string(index=False))

    return {'comparisons': dict(all_comparisons)}

if __name__ == "__main__":
    # Example usage
    file_paths = [
        "/content/drive/MyDrive/eval_outputs/results_claude-3-5-sonnet-20241022.csv",
        "/content/drive/MyDrive/eval_outputs/results_gemini-1.5-pro-002.csv", 
        "/content/drive/MyDrive/eval_outputs/results_gpt-4o-2024-08-06.csv"
    ]
    dfs = [load_and_clean_data(p) for p in file_paths]
    analyze_evaluation_results(dfs, file_paths)
