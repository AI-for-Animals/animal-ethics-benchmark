import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import glob

def load_and_clean_data(file_path):
    """Load CSV and convert all score columns to numeric."""
    df = pd.read_csv(file_path)
    # Get all score columns
    score_cols = [c for c in df.columns if c.endswith('_score')]
    # Convert all score columns to numeric
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
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

def load_model_data(model_files):
    """Load and combine data for each model, handling multiple files if present."""
    model_data = {}
    observations_per_model = {}
    
    for model, files in model_files.items():
        if len(files) == 1:
            # Single file case
            df = load_and_clean_data(files[0])
            score_cols = [c for c in df.columns if c.endswith('_score')]
            # Average across all score columns
            df['score'] = df[score_cols].mean(axis=1)
            total_obs = len(df)
        else:
            # Multiple files case
            all_scores = []
            for f in files:
                df = load_and_clean_data(f)
                score_cols = [c for c in df.columns if c.endswith('_score')]
                file_scores = df[score_cols].mean(axis=1)
                all_scores.append(file_scores)
            
            # Keep metadata from first file
            df = load_and_clean_data(files[0])
            # Average scores across files and score columns
            df['score'] = pd.concat(all_scores, axis=1).mean(axis=1)
            total_obs = len(df) * len(files)
        
        model_data[model] = df
        observations_per_model[model] = total_obs
    
    return model_data, observations_per_model

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
    Primary analysis function for non-scoring models.
    """
    # Get all relevant files for each model
    model_files = get_model_files(base_path, models)
    
    # Load and combine data for each model
    model_data, observations_per_model = load_model_data(model_files)
    
    def get_arrays():
        """Return (score_arrays, tags_arrays)."""
        score_arrays = [df['score'].values for df in model_data.values()]
        tags_arrays = [df['tag3'].values if 'tag3' in df.columns 
                      else np.array(['NA']*len(df)) for df in model_data.values()]
        return score_arrays, tags_arrays
    
    def format_comparisons(model_pairs, paired, clustered):
        score_arrays, tags_arrays = get_arrays()
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
                'Mean M1': np.mean(dA),
                'Mean M2': np.mean(dB)
            }
            row.update(stats)
            results.append(row)
        return pd.DataFrame(results)
    
    # Generate comparison tables
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
        df = format_comparisons(model_pairs, paired=p, clustered=c)
        all_comparisons.append((desc, df))
    
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
        "claude-3-5-haiku-20241022",
        "gemini-1.5-flash-002",
        "gpt-4o-mini-2024-07-18"
    ]
    
    analyze_evaluation_results(BASE_PATH, MODELS)