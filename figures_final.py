import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

def parse_score(score_str):
    """Extract score and confidence intervals from string like '0.0223 (0.0206, 0.0241)'"""
    pattern = r'(-?\d+\.\d+)\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)'
    match = re.match(pattern, score_str)
    if match:
        score = float(match.group(1))
        ci_low = float(match.group(2))
        ci_high = float(match.group(3))
        return score, ci_low, ci_high
    return None, None, None

def get_model_color(name):
    """Return color based on model name"""
    colors = {
        'claude': '#4A90E2',
        'gpt': '#45B764',
        'gemini': '#E6A040',
        'mistral': '#9B6B9E',
        'deepseek': '#E57373'
    }
    return next((v for k, v in colors.items() if k in name.lower()), '#757575')

def create_model_chart(data, title, filename, figsize=(4, 4)):
    """Create bar chart for given model data"""
    plt.figure(figsize=figsize)
    
    # Extract data
    scores = []
    yerr_low = []
    yerr_high = []
    names = []
    colors = []
    
    for _, row in data.iterrows():
        score, ci_low, ci_high = parse_score(row['Score'])
        if score is not None:
            scores.append(score)
            yerr_low.append(score - ci_low)
            yerr_high.append(ci_high - score)
            names.append(row['Model'])
            colors.append(get_model_color(row['Model']))
    
    # Create plot background with grid
    plt.grid(True, axis='y', alpha=0.3, zorder=0)  # Grid goes behind
    
    # Add black gridline at zero
    plt.axhline(y=0, color='black', linewidth=1.0, zorder=2)  # Thicker black line at zero
    
    # Create bar plot
    x = np.arange(len(names))
    bars = plt.bar(x, scores, color=colors, width=0.8, zorder=3)  # Bars above baseline
    
    # Add error bars on top
    plt.errorbar(x, scores, yerr=[yerr_low, yerr_high], fmt='none', color='gray', capsize=5, zorder=4)
    
    # Format axes
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Score')
    
    # Set y-axis limits
    # plt.ylim(-0.04, 0.1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Hardcoded data from the CSV
    data = {
        'Model': [
            'Claude-3.5-Sonnet',
            'Gemini-1.5-Pro',
            'GPT-4o',
            'Claude-3.5-Haiku',
            'Gemini-1.5-Flash',
            'GPT-4o-Mini',
            'DeepSeek-v3',
            'Llama-3.3',
            'Mistral-Large'
        ],
        'Score': [
            '0.0223 (0.0206, 0.0241)',
            '0.0634 (0.0621, 0.0646)',
            '0.0146 (0.0132, 0.0160)',
            '0.0147 (-0.0016, 0.0310)',
            '0.0460 (0.0296, 0.0624)',
            '0.0053 (-0.0110, 0.0216)',
            '0.0410 (0.0234, 0.0586)',
            '-0.0098 (-0.0277, 0.0081)',
            '0.0701 (0.0527, 0.0875)'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create three separate charts
    # Large models
    large_models = df[df['Model'].isin(['Claude-3.5-Sonnet', 'Gemini-1.5-Pro', 'GPT-4o'])]
    create_model_chart(large_models, 'Large Language Models', 'large_models.png')
    
    # Compact models
    compact_models = df[df['Model'].isin(['Claude-3.5-Haiku', 'Gemini-1.5-Flash', 'GPT-4o-Mini'])]
    create_model_chart(compact_models, 'Compact Models', 'compact_models.png')
    
    # Open models
    open_models = df[df['Model'].isin(['DeepSeek-v3', 'Llama-3.3', 'Mistral-Large'])]
    create_model_chart(open_models, 'Open Models', 'open_models.png')

if __name__ == '__main__':
    main()