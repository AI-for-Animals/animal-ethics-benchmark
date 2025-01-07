import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import ast
from scipy.stats import norm
from collections import defaultdict

#
# -------------------- FIRST PART: DATA LOADING & CLEANUP --------------------
#

def load_data_with_multiindex(file_paths):
    """
    Reads CSVs with a multi-index header (two header rows).
    Returns a list of DataFrames.
    """
    # Each CSV is read with header=[0,1], producing a multi-level column index
    dataframes = [pd.read_csv(path, header=[0, 1]) for path in file_paths]
    return dataframes

def flatten_multiindex_columns(df):
    """
    Flattens the multi-level columns into a single level by dropping
    the first level or combining them. Here we drop the first level
    and keep only the second (since the second snippet typically expects
    single-level columns like 'score', 'tags', etc.).
    """
    # If you prefer to combine both levels, you can do:
    # df.columns = ['_'.join(col).strip() for col in df.columns.to_flat_index()]
    # Instead, we will just drop the first level:
    df.columns = df.columns.droplevel(0)
    return df

def parse_tags_column(df):
    """
    Parse the 'tags' column from string representations of lists into actual lists or single tags.
    Assumes the relevant column is literally named 'tags' after flattening.
    """
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(
            lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith("[") else x
        )
    return df

def clean_and_convert_scores(df):
    """
    Convert any column whose name contains 'score' into numeric.
    """
    for col in df.columns:
        if 'score' in col.lower():
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


#
# -------------------- SECOND PART: ANALYSIS (FOUR COMPARISON TABLES) --------------------
#

def get_model_name(path): 
    """A helper to infer model names from file path."""
    path_lower = path.lower()
    if 'claude' in path_lower:
        return 'Claude'
    elif 'gpt' in path_lower:
        return 'GPT'
    else:
        return 'Gemini'


def compute_cluster_metrics(dfs, models, tags_column='tags'):
    """
    Compute mean and SD for each cluster across all models.
    Expects each df to have a column named 'tags' and one or more score columns.
    The count values will be divided by 3 in the final output.
    """
    cluster_stats = defaultdict(lambda: {'scores': [], 'count': 0})
    
    for df, model in zip(dfs, models):
        if tags_column not in df.columns:
            continue
        
        # Identify any columns that have 'score' in their name
        score_cols = [c for c in df.columns if 'score' in c.lower()]
        
        for tag in df[tags_column].fillna('uncategorized').unique():
            mask = df[tags_column].fillna('uncategorized') == tag
            scores = df.loc[mask, score_cols].mean(axis=1).values
            cluster_stats[tag]['scores'].extend(scores)
            cluster_stats[tag]['count'] += len(scores)
    
    results = []
    for tag, data in cluster_stats.items():
        scores = np.array(data['scores'])
        if len(scores) == 0:
            continue
            
        # Divide the count by 3 before adding to results
        adjusted_count = data['count'] // 3  # Using integer division
        
        results.append({
            'Cluster': tag,
            'Count': adjusted_count,  # This will now be the divided count
            'Mean Score': np.mean(scores),
            'SD': np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        })
    
    # Sort by descending count (now using the divided values)
    return pd.DataFrame(results).sort_values('Count', ascending=False)


#
# ------------- STATISTICAL COMPARISONS: 4 TABLES (with/without pairing/cluster) -------------
#

def compute_paired_clustered_comparison(data_A, data_B, tags):
    """
    Paired + clustered:
      * difference in means
      * standard error using the question-level difference with cluster corrections
      * 95% CI
      * p-value
    """
    n = len(data_A)
    diff = data_A - data_B
    clusters = pd.Series(tags).fillna('uncategorized')
    d_bar = np.mean(diff)
    
    # Collect within-cluster sums of (d_i - d_bar)*(d_j - d_bar)
    cluster_terms = []
    for cluster in clusters.unique():
        mask = clusters == cluster
        # If multiple items in that cluster, gather outer products
        if mask.sum() > 1:
            cluster_diff = diff[mask]
            demeaned = cluster_diff - d_bar
            # Outer product sum
            cluster_terms.append(np.outer(demeaned, demeaned))
    
    if len(cluster_terms) > 0:
        cluster_sum = sum(t.sum() for t in cluster_terms)
        se_paired_clustered = np.sqrt(cluster_sum / (n ** 2))
    else:
        # fallback to naive paired if all singletons or no valid tags
        var_diff = np.var(diff, ddof=1)
        se_paired_clustered = np.sqrt(var_diff / n)
    
    ci_low = d_bar - 1.96 * se_paired_clustered
    ci_high = d_bar + 1.96 * se_paired_clustered
    z_val = d_bar / se_paired_clustered if se_paired_clustered != 0 else 0.0
    p_val = 2 * (1 - norm.cdf(abs(z_val)))
    
    return {
        'Mean Difference': d_bar,
        'SE': se_paired_clustered,
        '95% CI': (ci_low, ci_high),
        'p-value': p_val
    }

def compute_unpaired_unclustered_comparison(data_A, data_B):
    """
    Neither paired nor clustered: standard 2-sample difference ignoring correlation.
    """
    mean_A = np.mean(data_A)
    mean_B = np.mean(data_B)
    diff = mean_A - mean_B

    varA = np.var(data_A, ddof=1)
    varB = np.var(data_B, ddof=1)
    nA = len(data_A)
    nB = len(data_B)
    
    # Naive standard error for 2-sample difference:
    se = np.sqrt(varA/nA + varB/nB)
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    z_val = diff / se if se != 0 else 0.0
    p_val = 2 * (1 - norm.cdf(abs(z_val)))
    
    return {
        'Mean Difference': diff,
        'SE': se,
        '95% CI': (ci_low, ci_high),
        'p-value': p_val
    }

def compute_paired_unclustered_comparison(data_A, data_B):
    """
    Paired, unclustered: difference on each question, standard error ignoring clusters.
    """
    diff = data_A - data_B
    d_bar = np.mean(diff)
    var_diff = np.var(diff, ddof=1)
    n = len(diff)
    se = np.sqrt(var_diff / n)
    
    ci_low = d_bar - 1.96 * se
    ci_high = d_bar + 1.96 * se
    z_val = d_bar / se if se != 0 else 0.0
    p_val = 2 * (1 - norm.cdf(abs(z_val)))
    
    return {
        'Mean Difference': d_bar,
        'SE': se,
        '95% CI': (ci_low, ci_high),
        'p-value': p_val
    }

def compute_unpaired_clustered_comparison(data_A, data_B, tags):
    """
    Unpaired, clustered: treat A-scores and B-scores as separate sets,
    compute difference in means with a cluster-level correction.
    """
    df_tmp = pd.DataFrame({
        'score_A': data_A,
        'score_B': data_B,
        'cluster': pd.Series(tags).fillna('uncategorized')
    })
    # Compute cluster-level means
    cluster_stats = df_tmp.groupby('cluster').agg({
        'score_A': ['mean','count'],
        'score_B': ['mean','count']
    })
    cluster_stats.columns = ['A_mean','A_count','B_mean','B_count']
    cluster_stats = cluster_stats.reset_index()
    
    # difference in cluster means
    cluster_stats['d_c'] = cluster_stats['A_mean'] - cluster_stats['B_mean']
    d_bar = cluster_stats['d_c'].mean()
    # Variance across clusters
    diff_vals = cluster_stats['d_c'].values
    cvar = np.var(diff_vals, ddof=1)
    C = len(diff_vals)

    # standard error
    if C > 1:
        se = np.sqrt(cvar / C)
    else:
        # fallback if 1 cluster
        se = np.sqrt(np.var(data_A - data_B, ddof=1)/len(data_A))
    
    ci_low = d_bar - 1.96 * se
    ci_high = d_bar + 1.96 * se
    z_val = d_bar / se if se != 0 else 0.0
    p_val = 2 * (1 - norm.cdf(abs(z_val)))
    
    return {
        'Mean Difference': d_bar,
        'SE': se,
        '95% CI': (ci_low, ci_high),
        'p-value': p_val
    }


#
# -------------------- MAIN ANALYSIS WRAPPER --------------------
#

def analyze_evaluation_results(dfs, file_paths):
    """
    Given the *already loaded and cleaned* DataFrames and corresponding file_paths,
    1) Infers model names,
    2) Gathers 'tags' (if present),
    3) Prepares score arrays for each model,
    4) Computes cluster metrics,
    5) Computes 4 difference tables,
    6) Returns all results as a dict (and also prints them).
    """

    models = []
    tags_arrays = []
    score_arrays = []
    
    # Identify which columns are 'score' columns
    # We'll take the average of all 'score' columns as the model's numeric data
    for df, path in zip(dfs, file_paths):
        mname = get_model_name(path)
        models.append(mname)

        # If there's a 'tags' column, store it; otherwise store None
        if 'tags' in df.columns:
            tags_arrays.append(df['tags'].values)
        else:
            tags_arrays.append(None)

        # Identify the score columns for this model
        score_cols = [c for c in df.columns if 'score' in c.lower()]
        if len(score_cols) == 0:
            # If no score columns exist, append None
            score_arrays.append(None)
        else:
            # Compute the row-wise average across all score columns
            score_arrays.append(df[score_cols].mean(axis=1).values)


    # --- 1) Compute cluster metrics (mean & SD per cluster), if possible
    cluster_metrics = compute_cluster_metrics(dfs, models)  # uses df['tags'] + any 'score' columns

    # --- 2) Build the 4 comparison tables among pairs of models
    neither_list = []
    pairwise_list = []
    clustered_list = []
    pairwise_clustered_list = []
    
    n_models = len(models)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Score arrays for the two models
            dataA = score_arrays[i]
            dataB = score_arrays[j]
            # We'll use the i-th model's tags for the cluster-based corrections:
            tags_for_A = tags_arrays[i] if tags_arrays[i] is not None else None
            
            # 1) Neither pairwise nor clustered
            neither_res = compute_unpaired_unclustered_comparison(dataA, dataB)
            # 2) Pairwise (unclustered)
            pairwise_res = compute_paired_unclustered_comparison(dataA, dataB)
            # 3) Clustered (unpaired)
            if tags_for_A is not None:
                clust_res = compute_unpaired_clustered_comparison(dataA, dataB, tags_for_A)
            else:
                clust_res = neither_res
            # 4) Pairwise + clustered
            if tags_for_A is not None:
                pc_res = compute_paired_clustered_comparison(dataA, dataB, tags_for_A)
            else:
                pc_res = pairwise_res
            
            # Summarize
            neither_list.append({
                'Model 1': models[i],
                'Model 2': models[j],
                'Mean M1': round(np.mean(dataA),4),
                'Mean M2': round(np.mean(dataB),4),
                'Mean Difference': round(neither_res['Mean Difference'],4),
                'SE': round(neither_res['SE'],4),
                '95% CI': (
                    round(neither_res['95% CI'][0],4),
                    round(neither_res['95% CI'][1],4)
                ),
                'p-value': round(neither_res['p-value'],4)
            })
            
            pairwise_list.append({
                'Model 1': models[i],
                'Model 2': models[j],
                'Mean M1': round(np.mean(dataA),4),
                'Mean M2': round(np.mean(dataB),4),
                'Mean Difference': round(pairwise_res['Mean Difference'],4),
                'SE': round(pairwise_res['SE'],4),
                '95% CI': (
                    round(pairwise_res['95% CI'][0],4),
                    round(pairwise_res['95% CI'][1],4)
                ),
                'p-value': round(pairwise_res['p-value'],4)
            })
            
            clustered_list.append({
                'Model 1': models[i],
                'Model 2': models[j],
                'Mean M1': round(np.mean(dataA),4),
                'Mean M2': round(np.mean(dataB),4),
                'Mean Difference': round(clust_res['Mean Difference'],4),
                'SE': round(clust_res['SE'],4),
                '95% CI': (
                    round(clust_res['95% CI'][0],4),
                    round(clust_res['95% CI'][1],4)
                ),
                'p-value': round(clust_res['p-value'],4)
            })
            
            pairwise_clustered_list.append({
                'Model 1': models[i],
                'Model 2': models[j],
                'Mean M1': round(np.mean(dataA),4),
                'Mean M2': round(np.mean(dataB),4),
                'Mean Difference': round(pc_res['Mean Difference'],4),
                'SE': round(pc_res['SE'],4),
                '95% CI': (
                    round(pc_res['95% CI'][0],4),
                    round(pc_res['95% CI'][1],4)
                ),
                'p-value': round(pc_res['p-value'],4)
            })
    
    # Convert to DataFrame
    df_neither = pd.DataFrame(neither_list)
    df_pairwise = pd.DataFrame(pairwise_list)
    df_clustered = pd.DataFrame(clustered_list)
    df_pairwise_clustered = pd.DataFrame(pairwise_clustered_list)
    
    #
    # Print them out
    #
    print("\n=== Cluster Metrics ===")
    if cluster_metrics is not None and not cluster_metrics.empty:
        print(cluster_metrics)
    else:
        print("No clustering information available.")
    
    print("\n=== (1) Neither Pairwise nor Clustered ===")
    print(df_neither)
    
    print("\n=== (2) Pairwise (Unclustered) ===")
    print(df_pairwise)
    
    print("\n=== (3) Clustered (Unpaired) ===")
    print(df_clustered)
    
    print("\n=== (4) Pairwise-and-Clustered ===")
    print(df_pairwise_clustered)
    
    return {
        'cluster_metrics': cluster_metrics,
        'neither': df_neither,
        'pairwise': df_pairwise,
        'clustered': df_clustered,
        'pairwise_clustered': df_pairwise_clustered
    }


#
# -------------------- MAIN SCRIPT ENTRY POINT --------------------
#

if __name__ == "__main__":
    file_paths = [
        "/content/drive/MyDrive/eval_outputs/claude-3-5-haiku-20241022/combined_results_claude-3-5-haiku-20241022_cleaned_with_tags.csv",
        "/content/drive/MyDrive/eval_outputs/gpt-4o-mini-2024-07-18/combined_results_gpt-4o-mini-2024-07-18_cleaned_with_tags.csv",
        "/content/drive/MyDrive/eval_outputs/gemini-2.0-flash-exp/combined_results_gemini-2.0-flash-exp_cleaned_with_tags.csv"
    ]
    
    # 1) Load data with multi-index
    raw_dataframes = load_data_with_multiindex(file_paths)
    
    # 2) Flatten columns and clean each dataframe
    cleaned_dfs = []
    for df in raw_dataframes:
        df = flatten_multiindex_columns(df)
        df = parse_tags_column(df)
        df = clean_and_convert_scores(df)
        cleaned_dfs.append(df)
    
    # 3) Perform the analysis (computes cluster metrics + the 4 comparison tables)
    results = analyze_evaluation_results(cleaned_dfs, file_paths)
