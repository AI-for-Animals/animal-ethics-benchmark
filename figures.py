import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import sem

def load_data(base_path, file_paths):
    return [pd.read_csv(Path(base_path) / path) for path in file_paths]

def plot_score_matrix(dfs, models):
    """
    Plots a heatmap of means and their 95% confidence intervals for each (Evaluated Model, Judge) pair.
    
    dfs:   list of DataFrames, one per 'Evaluated Model'
    models list of model names (strings), also used to find columns in each df (f"{model}_score").
    """
    
    z = 1.96  # ~95% from standard normal
    
    # Collect (mean, lower_CI, upper_CI) for each cell
    all_cells = []
    for df in dfs:
        row_cells = []
        for model in models:
            scores = df[f"{model}_score"].dropna()
            mean_ = scores.mean()
            se_ = sem(scores)
            lower_ = mean_ - z * se_
            upper_ = mean_ + z * se_
            row_cells.append((mean_, lower_, upper_))
        all_cells.append(row_cells)
        
    # Numeric matrix for color-coding
    matrix = np.array([[cell[0] for cell in row] for row in all_cells])
    
    # Build annotation text with increased spacing between mean and CI
    annot_text = [
        [
            f"{cell[0]:.3f}\n\n({cell[1]:.3f}, {cell[2]:.3f})"  # Added extra newline for spacing
            for cell in row
        ]
        for row in all_cells
    ]
    
    plt.figure(figsize=(12, 10))  # Slightly larger figure
    
    # Format model names for the axes
    model_labels = [model.split('/')[-1] for model in models]
    
    # Create heatmap with adjusted parameters
    ax = sns.heatmap(
        matrix,
        annot=annot_text,
        fmt="",
        cmap='RdYlGn',
        vmin=matrix.min(),
        vmax=matrix.max(),
        xticklabels=model_labels,
        yticklabels=model_labels,
        annot_kws={'size': 13},  # Adjusted font size
        cbar_kws={'label': 'Score'},  # Colorbar basic settings
        square=True  # Make cells square
    )
    
    # Adjust tick labels
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    # Bold axis labels with increased size
    plt.xlabel('LLM-as-a-Judge', fontsize=14, labelpad=15, weight='bold')
    plt.ylabel('Evaluated Model', fontsize=14, labelpad=15, weight='bold')
    
    # Adjust layout with more padding
    plt.tight_layout(pad=1.5)
    # Adjust colorbar label size after creating the heatmap
    colorbar = ax.collections[0].colorbar
    colorbar.ax.set_ylabel('Score', fontsize=12, weight='bold')
    
    plt.savefig('score_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distributions(dfs, models):
    bounds = np.linspace(-7/6, 7/6, 8)
    ticks = [-1, -0.67, -0.33, 0, 0.33, 0.67, 1]
    
    for i, (df, model) in enumerate(zip(dfs, models)):
        plt.figure(figsize=(8, 5))
        score_cols = [col for col in df.columns if col.endswith('_score')]
        scores = df[score_cols].mean(axis=1)
        
        plt.hist(scores, bins=bounds, edgecolor='black', color='steelblue', alpha=0.7)
        model_name = model.split("/")[-1]
        plt.title(f'Score Distribution: {model_name}')
        plt.xlabel('Question-Level Scores')
        plt.ylabel('Frequency')
        plt.xticks(ticks)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'histogram_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    base_path = "/content/drive/MyDrive/eval_outputs"
    models = ['anthropic/claude-3-5-sonnet-20241022', 
              'google/gemini-1.5-pro-002',
              'openai/gpt-4o-2024-08-06']
    
    file_paths = [f"results_{model.split('/')[-1]}.csv" for model in models]
    dfs = load_data(base_path, file_paths)
    
    plot_score_matrix(dfs, models)
    plot_score_distributions(dfs, models)
