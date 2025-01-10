import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def wilson_interval(mean, n, confidence=0.95):
    """
    Calculate Wilson score interval for a mean score with error handling
    """
    if n <= 0 or mean < 0 or mean > 1:
        return 0.0
    
    z = norm.ppf((1 + confidence) / 2)
    try:
        standard_error = mean * (1 - mean) / np.sqrt(n)
        denominator = 1 + (z**2/n)
        spread = z * np.sqrt(max(0, (mean * (1-mean) + z**2/(4*n)) / n)) / denominator
        return float(spread) if not np.isnan(spread) else 0.0
    except:
        return 0.0

def plot_cluster_metrics(df, tag_name, num_models):
    plot_df = df.sort_values('Total_mean', ascending=False)
    errors = [wilson_interval(row['Total_mean'], row['Count'] * num_models) 
              for _, row in plot_df.iterrows()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(plot_df)), 
                  plot_df['Total_mean'],
                  yerr=errors,
                  capsize=5,
                  error_kw={'elinewidth': 1, 'capthick': 1})
    
    plt.title(f'Cluster Performance for {tag_name}')
    plt.xlabel('Clusters')
    plt.ylabel('Total Mean Score')
    plt.xticks(range(len(plot_df)), plot_df['Cluster'], rotation=45, ha='right')
    
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + error + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.ylim(0, max(plot_df['Total_mean'] + pd.Series(errors)) * 1.15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'cluster_plot_{tag_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_model_name_from_file(filename):
    return filename.split('results_')[1].split('.csv')[0].split('-20')[0]

def calculate_cluster_metrics(base_path):
    file_names = [
        'results_claude-3-5-haiku-20241022.csv',
        'results_gemini-2.0-flash-exp.csv',
        'results_gpt-4o-mini-2024-07-18.csv'
    ]
    num_models = len(file_names)
    tag_columns = ['tag1', 'tag2', 'tag3', 'tag4']
    all_results = {tag: {} for tag in tag_columns}
    
    # Process each file
    for file_name in file_names:
        df = pd.read_csv(f"{base_path}/{file_name}")
        score_cols = [col.replace('_answer', '_score') for col in df.columns if col.endswith('_answer')]
        df['mean_score'] = df[score_cols].mean(axis=1)
        
        for tag_col in tag_columns:
            cluster_means = df.groupby(tag_col)['mean_score'].agg(['mean', 'count', 'std'])
            for cluster_name, stats in cluster_means.iterrows():
                cluster_name = 'NA' if pd.isna(cluster_name) else cluster_name
                all_results[tag_col].setdefault(cluster_name, {})[file_name] = {
                    'mean': stats['mean'],
                    'count': stats['count'],
                    'std': stats['std']
                }
    
    # Format results
    results = {}
    for tag_col in tag_columns:
        df_data = []
        for cluster, file_stats in all_results[tag_col].items():
            row = {'Cluster': cluster, 'Count': list(file_stats.values())[0]['count']}
            
            # Calculate total statistics
            means = [stats['mean'] for stats in file_stats.values()]
            row['Total_mean'] = np.mean(means)
            row['Total_std'] = np.sqrt(np.mean([stats['std']**2 for stats in file_stats.values()])) / np.sqrt(num_models)
            
            # Add individual model results
            for file_name in file_names:
                stats = file_stats[file_name]
                model_name = get_model_name_from_file(file_name)
                row[f'{model_name}_mean'] = stats['mean']
                row[f'{model_name}_std'] = stats['std']
            
            df_data.append(row)
        
        # Create DataFrame with ordered columns
        cols = ['Cluster', 'Count', 'Total_mean', 'Total_std']
        df = pd.DataFrame(df_data)
        other_cols = [col for col in df.columns if col not in cols]
        results[tag_col] = df[cols + other_cols].sort_values('Cluster', ascending=True)
        
        # Create plot with correct sample size
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
