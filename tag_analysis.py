"""
Tag Analysis Tool for Model Evaluation Results

Performs statistical comparison between two groups of tags in model evaluation data.
Each group can contain multiple tags combined with OR/AND logic.

Usage:
    python tag_analysis.py --tags1 tag1 [tag1_2 ...] --tags2 tag2 [tag2_2 ...] [--paired] [--combination {OR,AND}]

Arguments:
    --tags1: First group of tags (one or more)
    --tags2: Second group of tags (one or more)
    --paired: Use paired statistical test (default: unpaired)
    --combination: How to combine multiple tags within group (default: OR)
                  OR: samples matching any of the tags
                  AND: samples matching all of the tags

Example usage:
    Compare welfare-suggestive vs harm-suggestive:
    python tag_analysis.py --tags1 welfare-suggestive --tags2 harm-suggestive
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel
import argparse

def load_data(file_paths):
    """Load and combine data from all files."""
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        score_cols = [col for col in df.columns if col.endswith('_score')]
        df['mean_score'] = df[score_cols].mean(axis=1)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def get_tag_scores(df, tags, tag_columns=['tag1', 'tag2', 'tag3', 'tag4'], combination_type='OR'):
    """Get scores for specified tags."""
    if combination_type == 'OR':
        mask = pd.concat([df[col].isin(tags) for col in tag_columns], axis=1).any(axis=1)
    else:  # AND
        # Each row must have all tags distributed across its columns
        row_tags = df[tag_columns].values.tolist()
        mask = [set(tags).issubset(set(row)) for row in row_tags]
    return df.loc[mask, 'mean_score']

def compare_tag_groups(df, tags1, tags2, paired=False, combination_type='OR'):
    """Compare scores between two tag groups."""
    scores1 = get_tag_scores(df, tags1, combination_type=combination_type)
    scores2 = get_tag_scores(df, tags2, combination_type=combination_type)
    
    # Check if we have enough samples
    if len(scores1) == 0 or len(scores2) == 0:
        raise ValueError(f"No samples found for one or both tag groups. Group 1: {len(scores1)} samples, Group 2: {len(scores2)} samples")
    
    mean1, mean2 = scores1.mean(), scores2.mean()
    diff = mean1 - mean2
    
    if paired:
        if len(scores1) != len(scores2):
            raise ValueError("Paired test requires equal number of samples in both groups")
        t_stat, p_value = ttest_rel(scores1, scores2)
    else:
        t_stat, p_value = ttest_ind(scores1, scores2, equal_var=False)
    
    # Calculate 95% CI for the difference
    if paired:
        diff_scores = scores1.values - scores2.values  # Ensure numpy array
        se = np.std(diff_scores, ddof=1) / np.sqrt(len(diff_scores))
    else:
        se = np.sqrt(scores1.var(ddof=1)/len(scores1) + scores2.var(ddof=1)/len(scores2))
    
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se
    
    return {
        'Tags1': '+'.join(tags1),
        'Tags2': '+'.join(tags2),
        'Mean1': mean1,
        'Mean2': mean2,
        'Difference': diff,
        'SE': se,
        '95% CI': (ci_lower, ci_upper),
        'p-value': p_value,
        'N1': len(scores1),
        'N2': len(scores2)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare tag groups in model evaluations')
    parser.add_argument('--tags1', nargs='+', required=True, help='First group of tags')
    parser.add_argument('--tags2', nargs='+', required=True, help='Second group of tags')
    parser.add_argument('--paired', action='store_true', help='Use paired test')
    parser.add_argument('--combination', choices=['AND', 'OR'], default='OR', 
                       help='How to combine multiple tags within group')
    
    args = parser.parse_args()
    
    # Use the same file paths as in the original script
    base_path = "/content/drive/MyDrive/eval_outputs"
    file_paths = [f"{base_path}/results_{model}.csv" for model in 
                 ['claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'gpt-4o-2024-08-06']]
    
    df = load_data(file_paths)
    
    try:
        results = compare_tag_groups(df, args.tags1, args.tags2, args.paired, args.combination)
        
        # Print results in a formatted table
        print("\n=== Tag Group Comparison ===")
        print(f"Group 1: {results['Tags1']}")
        print(f"Group 2: {results['Tags2']}")
        print("\nResults:")
        print(f"{'Metric':<15} {'Value':<15}")
        print("-" * 30)
        print(f"Mean Group 1   {results['Mean1']:.4f} (n={results['N1']})")
        print(f"Mean Group 2   {results['Mean2']:.4f} (n={results['N2']})")
        print(f"Difference     {results['Difference']:.4f}")
        print(f"SE            {results['SE']:.4f}")
        print(f"95% CI        ({results['95% CI'][0]:.4f}, {results['95% CI'][1]:.4f})")
        print(f"p-value       {results['p-value']:.4f}")
    
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check that the specified tags exist in the data and try again.")
