import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(base_path, file_paths):
    return [pd.read_csv(Path(base_path) / path) for path in file_paths]

def plot_score_matrix(dfs, models):
    all_scores = []
    for i, df in enumerate(dfs):
        score_cols = [col for col in df.columns if col.endswith('_score')]
        means = df[score_cols].mean()
        all_scores.append([means[f"{model}_score"] for model in models])
    
    matrix = np.array(all_scores)
    plt.figure(figsize=(10, 8))
    
    # Using RdYlGn (Red-Yellow-Green) colormap
    # Red (0) -> Yellow (0.5) -> Green (1)
    sns.heatmap(matrix, annot=True, fmt='.4f', 
                cmap='RdYlGn',  
                vmin=matrix.min(), vmax=matrix.max(),
                xticklabels=models, yticklabels=models,
                annot_kws={'size': 10},
                cbar_kws={'label': 'Score'})
    
    plt.title('Model Score Matrix')
    plt.tight_layout()
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
    models = ['anthropic/claude-3-5-haiku-20241022', 
              'google/gemini-2.0-flash-exp',
              'openai/gpt-4o-mini-2024-07-18']
    
    file_paths = [f"results_{model.split('/')[-1]}.csv" for model in models]
    dfs = load_data(base_path, file_paths)
    
    plot_score_matrix(dfs, models)
    plot_score_distributions(dfs, models)
