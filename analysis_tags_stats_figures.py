import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def t_interval(mean, std, n, confidence=0.95):
    if n < 2:
        return 0.0
    try:
        t_crit = t.ppf((1 + confidence) / 2, df=n-1)
        se = std / np.sqrt(n)
        margin = t_crit * se
        return float(margin) if not np.isnan(margin) else 0.0
    except:
        return 0.0

def clean_label(label):
    label = str(label).strip('[]\'\"')
    if 'subreddit-' in label:
        label = label.replace('subreddit-', 'r/')
    if label.startswith('scenario-'):
        number = label.split('-')[1]
        label = f'S{number}'
    return label

def plot_cluster_metrics(df, tag_name, num_models):
    plot_df = df.sort_values('Total_mean', ascending=False)
    errors = [t_interval(row['Total_mean'], row['Total_std'], row['Count'] * num_models) 
              for _, row in plot_df.iterrows()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(plot_df)), 
            plot_df['Total_mean'],
            yerr=errors,
            capsize=5,
            error_kw={'elinewidth': 1, 'capthick': 1})
    
    plt.ylabel('')
    clean_labels = [clean_label(label) for label in plot_df['Cluster']]
    plt.xticks(range(len(plot_df)), 
               clean_labels,
               rotation=45,
               ha='right',
               fontsize=8)
    
    plt.xlabel('')
    min_val = min(plot_df['Total_mean'] - pd.Series(errors))
    max_val = max(plot_df['Total_mean'] + pd.Series(errors))
    ymin = min(0, min_val * 1.15)
    ymax = max_val * 1.15
    plt.ylim(ymin, ymax)
    plt.grid(False)
    plt.tight_layout()
    
    plt.savefig(f'cluster_plot_{tag_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_model_name_from_file(filename):
    return filename.split('results_')[1].split('.csv')[0].split('-20')[0]

def adjust_for_self_preference(scores_dict, judge_models):
    """
    Adjust scores to account for self-preference bias
    """
    adjusted_scores = {}
    
    for model, file_stats in scores_dict.items():
        if model not in judge_models:  # Non-judge models keep original scores
            adjusted_scores[model] = file_stats
            continue
            
        # For judge models, calculate adjustment
        other_judges = [j for j in judge_models if j != model]
        received_scores = [scores_dict[j]['mean'] for j in other_judges 
                         if j in scores_dict and not np.isnan(scores_dict[j]['mean'])]
        
        if received_scores:
            # Calculate h (how this model typically scores others)
            other_scores = []
            for other_model in other_judges:
                if other_model in scores_dict and not np.isnan(scores_dict[other_model]['mean']):
                    other_scores.append(scores_dict[other_model]['mean'])
            
            if other_scores:
                h = np.mean(other_scores)
                # Adjusted score calculation
                adj_score = (sum(received_scores) + h) / (len(other_judges) + 1)
                
                # Update stats with adjusted score
                adjusted_stats = file_stats.copy()
                adjusted_stats['mean'] = adj_score
                adjusted_scores[model] = adjusted_stats
            else:
                adjusted_scores[model] = file_stats
        else:
            adjusted_scores[model] = file_stats
            
    return adjusted_scores

def calculate_cluster_metrics(base_path):
    file_names = [
        'results_claude-3-5-sonnet-20241022_run1.csv',
        'results_claude-3-5-sonnet-20241022_run2.csv',
        'results_claude-3-5-sonnet-20241022_run3.csv',
        'results_claude-3-5-sonnet-20241022_run4.csv',
        'results_claude-3-5-sonnet-20241022_run5.csv',
        'results_gemini-1.5-pro-002_run1.csv',
        'results_gemini-1.5-pro-002_run2.csv',
        'results_gemini-1.5-pro-002_run3.csv',
        'results_gemini-1.5-pro-002_run4.csv',
        'results_gemini-1.5-pro-002_run5.csv',
        'results_gpt-4o-2024-08-06_run1.csv',
        'results_gpt-4o-2024-08-06_run2.csv',
        'results_gpt-4o-2024-08-06_run3.csv',
        'results_gpt-4o-2024-08-06_run4.csv',
        'results_gpt-4o-2024-08-06_run5.csv',
        'results_claude-3-5-haiku-20241022.csv',
        'results_gemini-1.5-flash-002.csv',
        'results_gpt-4o-mini-2024-07-18.csv',
        'results_DeepSeek-V3.csv',
        'results_Llama-3.3-70B-Instruct-Turbo.csv',
        'results_mistral-large-2411.csv'
    ]
    
    # Define judge models
    judge_models = [
        'claude-3-5-sonnet-20241022',
        'gemini-1.5-pro-002',
        'gpt-4o-2024-08-06'
    ]
    
    num_models = len(file_names)
    tag_columns = ['tag1', 'tag2', 'tag3', 'tag4']
    all_results = {tag: {} for tag in tag_columns}
    
    for file_name in file_names:
        df = pd.read_csv(f"{base_path}/{file_name}")
        score_cols = [col for col in df.columns if col.endswith('_score')]
        
        if not score_cols:
            print(f"No _score columns found for {file_name}. Skipping.")
            continue
            
        # Calculate mean scores before adjustment
        df['mean_score'] = df[score_cols].mean(axis=1)
        
        for tag_col in tag_columns:
            cluster_means = df.groupby(tag_col)['mean_score'].agg(['mean', 'count', 'std'])
            for cluster_name, stats in cluster_means.iterrows():
                cluster_name = 'NA' if pd.isna(cluster_name) else cluster_name
                model_name = get_model_name_from_file(file_name)
                
                all_results[tag_col].setdefault(cluster_name, {})[model_name] = {
                    'mean': stats['mean'],
                    'count': stats['count'],
                    'std': stats['std']
                }
    
    # Format results with self-preference adjustment
    results = {}
    for tag_col in tag_columns:
        df_data = []
        for cluster, model_stats in all_results[tag_col].items():
            # Adjust scores for self-preference
            adjusted_stats = adjust_for_self_preference(model_stats, judge_models)
            
            row = {'Cluster': cluster, 'Count': list(adjusted_stats.values())[0]['count']}
            
            # Calculate total statistics using adjusted scores
            means = [stats['mean'] for stats in adjusted_stats.values()]
            row['Total_mean'] = np.mean(means)
            row['Total_std'] = np.sqrt(np.mean([stats['std']**2 for stats in adjusted_stats.values()])) / np.sqrt(num_models)
            
            # Add individual model results
            for model_name in set([get_model_name_from_file(f) for f in file_names]):
                stats = adjusted_stats.get(model_name, {'mean': np.nan, 'std': np.nan})
                row[f'{model_name}_mean'] = stats['mean']
                row[f'{model_name}_std'] = stats['std']
            
            df_data.append(row)
        
        cols = ['Cluster', 'Count', 'Total_mean', 'Total_std']
        df = pd.DataFrame(df_data)
        other_cols = [col for col in df.columns if col not in cols]
        results[tag_col] = df[cols + other_cols].sort_values('Cluster', ascending=True)
        
        plot_cluster_metrics(results[tag_col], tag_col, num_models)
    
    return results

if __name__ == "__main__":
    base_path = "/content/drive/MyDrive/eval_outputs"
    all_tag_results = calculate_cluster_metrics(base_path)
    
    for tag_col, results_df in all_tag_results.items():
        print(f"\n\nResults for {tag_col}:")
        print("=" * 80)
        print(results_df.round(4).to_string(index=False))
        results_df.to_csv(f"cluster_metrics_{tag_col}.csv", index=False)
