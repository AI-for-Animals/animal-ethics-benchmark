import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Wilson Score Interval Calculation
def wilson_score_interval(p, n, z=1.96):
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - spread, center + spread

# Data from the image
models = [
    'claude-3-haiku-20240307',
    'claude-3-5-sonnet-20241022',
    'gpt-4o-2024-08-06',
    'gemini-1.5-pro-002',
    'Meta-Llama-3.1-405B-Instruct-Turbo',
    'o1-preview-2024-09-12',
    'mistral/mistral-large-2407'
]

scores = [0.619, 0.667, 0.786, 0.619, 0.690, 0.738, 0.762]
n_samples = 42  # From the image

# Sort by scores
pairs = list(zip(models, scores))
pairs.sort(key=lambda x: x[1], reverse=True)
models, scores = zip(*pairs)

# Calculate confidence intervals
confidence_intervals = []
for score in scores:
    lower, upper = wilson_score_interval(score, n_samples)
    confidence_intervals.append((upper - score, score - lower))
errors = np.array(confidence_intervals).T

# Create plot
plt.figure(figsize=(12, 6))
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.facecolor'] = '#f0f0f0'

x = np.arange(len(models))
bars = plt.bar(x, scores, color='#1f77b4', alpha=0.85)
plt.errorbar(x, scores, yerr=errors, fmt='none', color='#333333', capsize=10)

plt.ylim(0, 1.0)
plt.grid(True, axis='y', linestyle='--', color='#cccccc', alpha=0.7)

# Format x-axis labels
plt.xticks(x, models, rotation=45, ha='right', fontsize=10, weight='bold', color='#333333')
plt.yticks(fontsize=12, weight='bold', color='#333333')
plt.ylabel('Score', fontsize=14, weight='bold', color='#333333')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom',
             fontsize=12, weight='bold', color='#333333')

# Adjust layout
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()

plt.show()
