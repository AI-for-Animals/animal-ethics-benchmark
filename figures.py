import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(file_paths):
    return [pd.read_csv(path) for path in file_paths]

def plot_score_matrix(dfs, models, save_path):
    # Extract scores and calculate means
    all_scores = []
    for i, df in enumerate(dfs):
        score_cols = [col for col in df.columns if col.endswith('_score')]
        means = df[score_cols].mean()
        all_scores.append([means[f"{model}_score"] for model in models])
    
    # Create matrix
    matrix = np.array(all_scores)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='viridis',
                xticklabels=models, yticklabels=models)
    plt.title('Model Score Matrix')
    plt.tight_layout()
    plt.savefig(save_path / 'score_matrix.png')
    plt.close()

def plot_score_distributions(dfs, models, save_path):
    plt.figure(figsize=(15, 5))
    bounds = np.linspace(-7/6, 7/6, 8)
    ticks = [-1, -0.67, -0.33, 0, 0.33, 0.67, 1]
    
    for i, (df, model) in enumerate(zip(dfs, models)):
        score_cols = [col for col in df.columns if col.endswith('_score')]
        scores = df[score_cols].mean(axis=1)  # Average scores for each row in this file
        
        plt.subplot(1, 3, i+1)
        plt.hist(scores, bins=bounds, edgecolor='black')
        plt.title(f'Histogram of Raw Scores for {model.split("/")[-1]}')
        plt.xlabel('Question-Level Scores')
        plt.ylabel('Frequency')
        plt.xticks(ticks)
        
    plt.tight_layout()
    plt.savefig(save_path / 'score_distributions.png')
    plt.close()

if __name__ == "__main__":
    base_path = "/content/drive/MyDrive/eval_outputs"
    save_path = Path(base_path) / "figures"
    save_path.mkdir(exist_ok=True)
    
    models = ['anthropic/claude-3-5-haiku-20241022', 
              'google/gemini-2.0-flash-exp',
              'openai/gpt-4o-mini-2024-07-18']
    
    file_paths = [f"{base_path}/synthetic_{model.split('/')[-1]}.csv" for model in models]
    dfs = load_data(file_paths)
    
    plot_score_matrix(dfs, models, save_path)
    plot_score_distributions(dfs, models, save_path)
