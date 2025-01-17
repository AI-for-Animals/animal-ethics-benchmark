import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def t_interval(mean, std, n, confidence=0.95):
    """
    Calculate t-distribution based confidence interval
    Returns the margin of error (half-width of the CI)
    """
    if n < 2:  # Need at least 2 samples for std
        return 0.0
    
    try:
        # Get critical value from t-distribution
        t_crit = t.ppf((1 + confidence) / 2, df=n-1)
        # Calculate standard error
        se = std / np.sqrt(n)
        # Calculate margin of error
        margin = t_crit * se
        return float(margin) if not np.isnan(margin) else 0.0
    except:
        return 0.0

def plot_cluster_metrics(df, tag_name, num_models):
    plot_df = df.sort_values('Total_mean', ascending=False)
    # Calculate error margins using t-distribution
    errors = [t_interval(row['Total_mean'], row['Total_std'], row['Count'] * num_models) 
              for _, row in plot_df.iterrows()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(plot_df)), 
                  plot_df['Total_mean'],
                  yerr=errors,
                  capsize=5,
                  error_kw={'elinewidth': 1, 'capthick': 1})
    
    plt.ylabel('Total Mean Score')
    plt.xticks(range(len(plot_df)), plot_df['Cluster'], rotation=45, ha='right')
    
    # Add rotated value labels at same position
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + error + 0.01,
                f'{height:.3f}',
                ha='center', 
                va='bottom',
                rotation=45)
    
    # Adjust y-axis limits to ensure 0 is visible and full error bars are shown
    min_val = min(plot_df['Total_mean'] - pd.Series(errors))
    max_val = max(plot_df['Total_mean'] + pd.Series(errors))
    ymin = min(0, min_val * 1.15)  # Include 0 and allow negative values
    ymax = max_val * 1.15
    plt.ylim(ymin, ymax)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'cluster_plot_{tag_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_model_name_from_file(filename):
    return filename.split('results_')[1].split('.csv')[0].split('-20')[0]

def calculate_cluster_metrics(base_path):
    file_names = [
        'results_claude-3-5-sonnet-20241022.csv',
        'results_gemini-1.5-pro-002.csv',
        'results_gpt-4o-2024-08-06.csv'
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
