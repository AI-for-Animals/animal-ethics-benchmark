import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df[[c for c in df.columns if c.endswith('_score')]] = df.filter(regex='_score$').apply(pd.to_numeric, errors='coerce')
    return df

def get_model_names(df):
    return [col[:-6] for col in df.columns if col.endswith('_score')]

def plot_results_matrix(dfs, output_path):
    models = get_model_names(dfs[0])
    matrix = np.array([[df[f"{judge}_score"].mean() if f"{judge}_score" in df.columns else 0 
                       for judge in models] for df, _ in zip(dfs, models)])
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', vmin=0.25, vmax=0.7)
    plt.title('Results Matrix (Rows=Scored Models, Columns=Judges)')
    plt.xlabel('Judge'), plt.ylabel('Model')
    
    names = [m.split('/')[-1] for m in models]
    plt.yticks(np.arange(len(models)) + 0.5, [f'Model: {m}' for m in names], rotation=0)
    plt.xticks(np.arange(len(models)) + 0.5, [f'Judge: {m}' for m in names], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_path}/results_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_histograms(dfs, output_path):
    models = get_model_names(dfs[0])
    bounds, ticks = np.linspace(-7/6, 7/6, 8), [-1, -0.67, -0.33, 0, 0.33, 0.67, 1]
    
    for model in models:
        scores = np.concatenate([df[f"{model}_score"].values for df in dfs 
                               if f"{model}_score" in df.columns])
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=bounds, edgecolor='black')
        plt.title(f'Histogram of Raw Scores for {model.split("/")[-1]}')
        plt.xlabel('Question-Level Scores'), plt.ylabel('Frequency')
        plt.xticks(ticks)
        plt.tight_layout()
        plt.savefig(f'{output_path}/histogram_{model.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

def estimate_biases(dfs, models):
    scores = [df[f"{judge}_score"].mean() for df in dfs for judge in models 
             if f"{judge}_score" in df.columns]
    n_models = len(models)
    
    def objective(x):
        true_scores, judge_biases, self_biases = x[:n_models], x[n_models:2*n_models], x[2*n_models:]
        return np.array([score - (judge_biases[j] + true_scores[i] + (self_biases[j] if i == j else 0))
                        for i, group in enumerate([scores[i:i+3] for i in range(0, len(scores), 3)])
                        for j, score in enumerate(group)])
    
    x0 = np.concatenate([[0.5] * n_models, [0.0] * n_models, [0.1] * n_models])
    bounds = (np.concatenate([[0.3] * n_models, [-0.5] * n_models, [-0.2] * n_models]),
             np.concatenate([[0.8] * n_models, [0.5] * n_models, [0.3] * n_models]))
    
    result = least_squares(objective, x0, bounds=bounds)
    judge_biases = result.x[n_models:2*n_models] - np.mean(result.x[n_models:2*n_models])
    return result.x[:n_models], judge_biases, result.x[2*n_models:]

def compute_comparison(data_A, data_B, tags=None, paired=False, clustered=False, adjusted=False,
                      true_scores=None, judge_biases=None, self_biases=None, model_indices=None):
    d_bar = np.mean(data_A) - np.mean(data_B)
    if paired:
        diff = data_A - data_B
        base_var = np.var(diff, ddof=1) / len(diff)
    else:
        base_var = np.var(data_A, ddof=1)/len(data_A) + np.var(data_B, ddof=1)/len(data_B)

    if clustered:
        clusters = pd.Series(tags).fillna('uncategorized')
        if paired:
            by_cluster = pd.DataFrame({'diff': data_A - data_B, 'cluster': clusters})
            cluster_means = by_cluster.groupby('cluster')['diff'].agg(['mean', 'count'])
            between_var = np.var(cluster_means['mean'], ddof=1)
            se = np.sqrt(between_var / len(cluster_means) + base_var)
        else:
            by_cluster = pd.DataFrame({'A': data_A, 'B': data_B, 'cluster': clusters})
            cluster_means = by_cluster.groupby('cluster').agg({
                'A': 'mean',
                'B': 'mean'
            })
            cluster_diffs = cluster_means['A'] - cluster_means['B']
            se = np.sqrt(np.var(cluster_diffs, ddof=1) / len(cluster_diffs) + base_var)
    else:
        se = np.sqrt(base_var)

    ci = (d_bar - 1.96 * se, d_bar + 1.96 * se)
    p_value = 2 * (1 - norm.cdf(abs(d_bar / se))) if se != 0 else 0.0
    return d_bar, se, ci, p_value

def analyze_evaluation_results(dfs, file_paths):
    models = get_model_names(dfs[0])
    true_scores, judge_biases, self_biases = estimate_biases(dfs, models)
    
    def get_adjusted_data():
        adj_dfs = []
        for i, df in enumerate(dfs):
            adj_df = df.copy()
            for col in [c for c in df.columns if c.endswith('_score')]:
                judge_idx = models.index(col[:-6])
                adj_df[col] -= judge_biases[judge_idx]
                if judge_idx == i:
                    adj_df[col] -= self_biases[judge_idx]
            adj_dfs.append(adj_df)
        return adj_dfs

    def get_arrays(use_adjusted=False):
        source = get_adjusted_data() if use_adjusted else dfs
        return ([df.filter(regex='_score$').mean(axis=1).values for df in source],
                [df['tag1'].values for df in source])

    def format_comparisons(model_pairs, paired, clustered, adjusted=False):
        score_arrays, tags_arrays = get_arrays(adjusted)
        return pd.DataFrame([{
            'Model 1': models[i], 'Model 2': models[j],
            'Mean M1': true_scores[i] if adjusted else np.mean(score_arrays[i]),
            'Mean M2': true_scores[j] if adjusted else np.mean(score_arrays[j]),
            **dict(zip(['Mean Difference', 'SE', '95% CI', 'p-value'],
                    compute_comparison(score_arrays[i], score_arrays[j],
                                    tags_arrays[i] if clustered else None,
                                    paired, clustered, adjusted,
                                    true_scores, judge_biases, self_biases, (i, j))))
        } for i, j in model_pairs])

    comparison_types = [
        ('Neither Pairwise nor Clustered', False, False),
        ('Pairwise Unclustered', True, False),
        ('Clustered Unpaired', False, True),
        ('Pairwise-and-Clustered', True, True)
    ]
    
    model_pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
    all_comparisons = [(title, format_comparisons(model_pairs, paired, clustered, adj))
                      for title, paired, clustered in comparison_types
                      for adj in [False, True]]

    cluster_metrics = compute_cluster_metrics(dfs, models)
    print_results(cluster_metrics, models, judge_biases, self_biases, all_comparisons, dfs)
    
    return {'cluster_metrics': cluster_metrics, 
            **{t.split()[0].lower(): df for t, df in all_comparisons}}

def compute_cluster_metrics(dfs, models):
    cluster_stats = defaultdict(lambda: {'scores': [], 'count': 0})
    for df in dfs:
        score_cols = [c for c in df.columns if c.endswith('_score')]
        for tag in df['tag1'].fillna('uncategorized').unique():
            mask = df['tag1'].fillna('uncategorized') == tag
            scores = df.loc[mask, score_cols].mean(axis=1).values
            cluster_stats[tag]['scores'].extend(scores)
            cluster_stats[tag]['count'] += len(scores)
    
    return pd.DataFrame([{
        'Cluster': tag,
        'Count': data['count'] // 3,
        'Mean Score': np.mean(scores := np.array(data['scores'])),
        'SD': np.std(scores, ddof=1) if len(scores) > 1 else 0.0
    } for tag, data in cluster_stats.items() if data['scores']]).sort_values('Count', ascending=False)

def print_results(cluster_metrics, models, judge_biases, self_biases, all_comparisons, dfs):
    if not cluster_metrics.empty:
        print("\n=== Cluster Metrics ===")
        expanded_data = []
        per_model_scores = defaultdict(dict)
        cluster_counts = defaultdict(int)
        
        for df in dfs:
            for tag in df['tag1'].fillna('uncategorized').unique():
                mask = df['tag1'].fillna('uncategorized') == tag
                cluster_counts[tag] = len(df[mask]) // len(models)
                for model in models:
                    if f"{model}_score" in df.columns:
                        per_model_scores[tag][model] = df.loc[mask, f"{model}_score"].mean()

        raw_scores = {model: all_comparisons[0][1][
            all_comparisons[0][1]['Model 1'] == model]['Mean M1'].iloc[0] 
            if len(all_comparisons[0][1][all_comparisons[0][1]['Model 1'] == model]) > 0 
            else all_comparisons[0][1][all_comparisons[0][1]['Model 2'] == model]['Mean M2'].iloc[0]
            for model in models}

        for tag, scores in per_model_scores.items():
            expanded_data.append({
                'Cluster': tag,
                'Count': cluster_counts[tag],
                'Mean Score': np.mean([scores[m] for m in models]),
                'SD': np.std([scores[m] for m in models], ddof=1),
                **{model: scores[model] for model in models}
            })
        
        expanded_data.append({
            'Cluster': 'Total',
            'Count': sum(cluster_counts.values()),
            'Mean Score': np.mean(list(raw_scores.values())),
            'SD': np.std(list(raw_scores.values()), ddof=1),
            **raw_scores
        })
        
        df = pd.DataFrame(expanded_data)
        df = df.sort_values('Count', ascending=False)
        numeric_cols = ['Mean Score', 'SD'] + models
        print(df.assign(**{col: df[col].map('{:.6f}'.format) 
                          for col in numeric_cols}).to_string(index=False))

    score_format = '{: >50} {:.4f}'
    sections = [
        ("\n=== Raw Scores ===", [(m, raw_scores[m]) for m in models]),
        ("\n=== Estimated Biases ===\n\nJudge Biases", zip(models, judge_biases)),
        ("\nSelf Biases", zip(models, self_biases)),
        ("\n=== Adjusted Scores ===", [(m, 
            all_comparisons[1][1][all_comparisons[1][1]['Model 1'] == m]['Mean M1'].iloc[0] 
            if len(all_comparisons[1][1][all_comparisons[1][1]['Model 1'] == m]) > 0 
            else all_comparisons[1][1][all_comparisons[1][1]['Model 2'] == m]['Mean M2'].iloc[0])
            for m in models])
    ]
    
    for header, items in sections:
        print(header)
        for model, value in items:
            print(score_format.format(model, value))

    for header, results in [("Raw Results", [c for i, c in enumerate(all_comparisons) if i % 2 == 0]),
                          ("Debiased Results", [c for i, c in enumerate(all_comparisons) if i % 2 == 1])]:
        print(f"\n=== {header} ===")
        for title, df in results:
            print(f"\n--- {title} ---")
            print(df.assign(**{
                col: df[col].map('{:.4f}'.format) 
                for col in ['Mean M1', 'Mean M2', 'Mean Difference', 'SE', 'p-value']
            }).assign(**{
                '95% CI': df.apply(lambda r: f"({r['95% CI'][0]:.4f}, {r['95% CI'][1]:.4f})", axis=1)
            }).to_string(index=True))

if __name__ == "__main__":
    base_path = "/content/drive/MyDrive/eval_outputs"
    file_paths = [f"{base_path}/reddit_{model}.csv" for model in 
                 ['claude-3-5-haiku-20241022', 'gemini-2.0-flash-exp', 'gpt-4o-mini-2024-07-18']]
    dfs = [load_and_clean_data(path) for path in file_paths]
    plot_results_matrix(dfs, base_path)
    plot_score_histograms(dfs, base_path)
    analyze_evaluation_results(dfs, file_paths)
