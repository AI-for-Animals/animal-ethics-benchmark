import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares

def load_and_clean_data(file_path):
    """Load CSV and convert all *_score columns to numeric."""
    df = pd.read_csv(file_path)
    score_cols = [c for c in df.columns if c.endswith('_score')]
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
    return df

def get_model_names(df):
    """Extract model names from columns that end with '_score'."""
    return [col[:-6] for col in df.columns if col.endswith('_score')]

def estimate_biases(dfs, models):
    """
    Estimate each model’s ‘true score,’ judge_bias, and self_bias.
    Bounds keep them in plausible ranges.
    """
    # Flatten each file’s average for each model
    scores = []
    for df in dfs:
        for judge in models:
            col = f"{judge}_score"
            if col in df.columns:
                scores.append(df[col].mean())
    n_models = len(models)

    def objective(x):
        # x layout: [true_scores (n), judge_biases (n), self_biases (n)]
        true_scores = x[:n_models]
        judge_biases = x[n_models:2*n_models]
        self_biases = x[2*n_models:]
        chunked = [scores[i:i+n_models] for i in range(0, len(scores), n_models)]

        residuals = []
        for i, group in enumerate(chunked):
            for j, score in enumerate(group):
                predicted = true_scores[i] + judge_biases[j]
                if i == j:
                    predicted += self_biases[j]
                residuals.append(score - predicted)
        return np.array(residuals)

    x0 = np.concatenate([
        [0.5]*n_models,  # initial guess for true_scores
        [0.0]*n_models,  # initial guess for judge_biases
        [0.1]*n_models   # initial guess for self_biases
    ])
    bounds = (
        np.concatenate([[0.3]*n_models, [-0.5]*n_models, [-0.2]*n_models]),
        np.concatenate([[0.8]*n_models, [0.5]*n_models, [0.3]*n_models])
    )
    result = least_squares(objective, x0, bounds=bounds)

    # Center judge_biases
    true_scores = result.x[:n_models]
    judge_biases = result.x[n_models:2*n_models] - np.mean(result.x[n_models:2*n_models])
    self_biases = result.x[2*n_models:]
    return true_scores, judge_biases, self_biases

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
    Primary function:
      - estimate biases
      - produce 8 comparison tables (4 raw, 4 debiased)
      - print cluster counts for tag3
    """
    models = get_model_names(dfs[0])
    true_scores, judge_biases, self_biases = estimate_biases(dfs, models)

    # Build "adjusted" data by subtracting judge_bias & self_bias
    def get_adjusted_data():
        adjusted_dfs = []
        for i, df in enumerate(dfs):
            new_df = df.copy()
            for col in [c for c in df.columns if c.endswith('_score')]:
                j_idx = models.index(col[:-6])
                new_df[col] -= judge_biases[j_idx]
                if j_idx == i:
                    new_df[col] -= self_biases[j_idx]
            adjusted_dfs.append(new_df)
        return adjusted_dfs

    def get_arrays(use_adjusted=False):
        """
        Return (score_arrays, tags_arrays).
         - score_arrays[i] = array of means across each row's _score columns
         - tags_arrays[i] = array of cluster tags from 'tag3'
        """
        data_source = get_adjusted_data() if use_adjusted else dfs
        score_arrays = [df.filter(regex='_score$').mean(axis=1).values for df in data_source]
        tags_arrays = [df['tag3'].values if 'tag3' in df.columns else np.array(['NA']*len(df))
                       for df in data_source]
        return score_arrays, tags_arrays

    def format_comparisons(model_pairs, paired, clustered, adjusted_label):
        """
        For each pair (i, j), compute the difference (raw or debiased).
        """
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
                row['Mean M1'] = true_scores[i]
                row['Mean M2'] = true_scores[j]
            else:
                row['Mean M1'] = np.mean(dA)
                row['Mean M2'] = np.mean(dB)
            row.update(stats)
            results.append(row)
        return pd.DataFrame(results)

    # 4 raw comparisons, then 4 debiased
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

    # Print everything
    print_results(models, judge_biases, self_biases, all_comparisons, dfs)

    # Return dictionary if you need it
    return {'comparisons': dict(all_comparisons)}

def print_results(models, judge_biases, self_biases, all_comparisons, dfs):
    """
    Print:
      - Observations per cluster (from 'tag3')
      - Raw scores (from the first Raw table)
      - Judge / self biases
      - Adjusted scores (from the first Debiased table)
      - Then print all 8 comparison tables
    """
    # 1) Print cluster counts for tag3
    print_cluster_counts(dfs)

    # 2) Build dictionary of raw_scores from the first "Neither Pairwise nor Clustered (Raw)"
    first_raw_title = "Neither Pairwise nor Clustered (Raw)"
    first_table = next((df for (title, df) in all_comparisons if title == first_raw_title), pd.DataFrame())
    raw_scores = {}
    for m in models:
        df_m1 = first_table[first_table['Model 1'] == m]
        if not df_m1.empty:
            raw_scores[m] = df_m1['Mean M1'].iloc[0]
        else:
            df_m2 = first_table[first_table['Model 2'] == m]
            if not df_m2.empty:
                raw_scores[m] = df_m2['Mean M2'].iloc[0]

    print("\n=== Raw Scores (Unadjusted) ===")
    for m in models:
        if m in raw_scores:
            print(f"{m:>40} {raw_scores[m]:.4f}")

    # 3) Judge & self biases
    print("\n=== Estimated Biases ===")
    print("\nJudge Biases:")
    for m, b in zip(models, judge_biases):
        print(f"{m:>40} {b:.4f}")

    print("\nSelf Biases:")
    for m, b in zip(models, self_biases):
        print(f"{m:>40} {b:.4f}")

    # 4) Adjusted scores from "Neither Pairwise nor Clustered (Debiased)"
    first_debiased_title = "Neither Pairwise nor Clustered (Debiased)"
    first_deb_table = next((df for (title, df) in all_comparisons if title == first_debiased_title), pd.DataFrame())
    adjusted_scores = {}
    for m in models:
        df_m1 = first_deb_table[first_deb_table['Model 1'] == m]
        if not df_m1.empty:
            adjusted_scores[m] = df_m1['Mean M1'].iloc[0]
        else:
            df_m2 = first_deb_table[first_deb_table['Model 2'] == m]
            if not df_m2.empty:
                adjusted_scores[m] = df_m2['Mean M2'].iloc[0]

    print("\n=== Adjusted Scores ===")
    for m in models:
        if m in adjusted_scores:
            print(f"{m:>40} {adjusted_scores[m]:.4f}")

    # 5) Print all 8 comparison tables
    print("\n=== Comparison Tables ===\n(N is # data points; #Clusters if clustered)")
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

def print_cluster_counts(dfs):
    """
    Show how many observations fall under each cluster in 'tag3'
    across all files combined.
    """
    all_tags = []
    for df in dfs:
        if 'tag3' in df.columns:
            all_tags.extend(df['tag3'].fillna('uncategorized').tolist())
        else:
            all_tags.extend(["NA"] * len(df))

    cluster_series = pd.Series(all_tags)
    cluster_counts = cluster_series.value_counts().reset_index()
    cluster_counts.columns = ['ClusterTag', 'Count']

    print("\n=== Observations per Cluster (tag3) ===")
    print(cluster_counts.to_string(index=False))

# -------------
# Example usage:
if __name__ == "__main__":
    base_path = "/content/drive/MyDrive/eval_outputs"  # Not used, but shown for context
    file_paths = [
        f"{base_path}/results_{m}.csv"
        for m in [
            'claude-3-5-haiku-20241022',
            'gemini-2.0-flash-exp',
            'gpt-4o-mini-2024-07-18'
        ]
    ]
    # Load data
    dfs = [load_and_clean_data(p) for p in file_paths]
    # Run main analysis
    analyze_evaluation_results(dfs, file_paths)
