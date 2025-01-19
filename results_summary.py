import pandas as pd
import glob
import os
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def process_evaluation_files(directory_path):
    file_pattern = os.path.join(directory_path, 'results_*.csv')
    files = glob.glob(file_pattern)
    results = {}
    confidence_intervals = {}
    n_obs = 4350  # base number of observations per score
    
    for file in files:
        df = pd.read_csv(file)
        score_columns = [col for col in df.columns if col.endswith('_score')]
        
        # Calculate means and CIs for each column
        file_results = {}
        file_cis = {}
        for col in score_columns:
            mean = df[col].mean()
            std_err = df[col].std() / np.sqrt(n_obs)  # individual scores
            ci_lower = mean - (1.96 * std_err)
            ci_upper = mean + (1.96 * std_err)
            
            file_results[col] = mean
            file_cis[col] = (ci_lower, ci_upper)
        
        # Calculate row average (across 3 scores) and its CI
        avg_mean = np.mean(list(file_results.values()))
        combined_scores = np.array([df[col] for col in score_columns])
        avg_std = np.std(np.mean(combined_scores, axis=0))
        # Use sqrt(3*n_obs) for row averages as we have 3 times more observations
        avg_ci_lower = avg_mean - (1.96 * avg_std / np.sqrt(3 * n_obs))
        avg_ci_upper = avg_mean + (1.96 * avg_std / np.sqrt(3 * n_obs))
        
        file_results['average'] = avg_mean
        file_cis['average'] = (avg_ci_lower, avg_ci_upper)
        
        filename = os.path.basename(file)
        filename = filename.replace('results_', '').replace('.csv', '')
        results[filename] = file_results
        confidence_intervals[filename] = file_cis
    
    if results:
        results_df = pd.DataFrame(results).T
        ci_df = pd.DataFrame(confidence_intervals).T
        
        new_columns = {col: col.split('/')[-1].replace('_score', '').strip() 
                      for col in results_df.columns if col != 'average'}
        new_columns['average'] = 'Average Score'
        results_df = results_df.rename(columns=new_columns)
        ci_df = ci_df.rename(columns=new_columns)
        
        # Format scores with CIs
        formatted_df = pd.DataFrame(index=results_df.index)
        for col in results_df.columns:
            formatted_df[col] = results_df.apply(
                lambda row: f"{row[col]:.3f} ({ci_df.loc[row.name, col][0]:.3f}, {ci_df.loc[row.name, col][1]:.3f})",
                axis=1
            )
        
        # Calculate and format average row (LLM-as-Judge AVERAGE SCORE)
        avg_scores = results_df.mean()
        avg_row_cis = {}
        for col in results_df.columns:
            if col != 'Average Score':
                # For judge columns in average row, use n=4350*9 as we're averaging across all observations
                std_err = results_df[col].std() / np.sqrt(9 * n_obs)
                ci_lower = avg_scores[col] - (1.96 * std_err)
                ci_upper = avg_scores[col] + (1.96 * std_err)
                avg_row_cis[col] = f"{avg_scores[col]:.3f} ({ci_lower:.3f}, {ci_upper:.3f})"
            else:
                # For the overall average (bottom-right cell), use n=4350*9
                std_err = results_df[col].std() / np.sqrt(9 * n_obs)
                ci_lower = avg_scores[col] - (1.96 * std_err)
                ci_upper = avg_scores[col] + (1.96 * std_err)
                avg_row_cis[col] = f"{avg_scores[col]:.3f} ({ci_lower:.3f}, {ci_upper:.3f})"
        
        avg_row = pd.DataFrame([avg_row_cis], index=['LLM-as-Judge AVERAGE SCORE'])
        formatted_df = pd.concat([formatted_df, avg_row])
        
        title = "LLM-as-a-Judge scores. 95% c.i. in brackets. n=4350 observations for each score (13050 per average score)"
        print(f"\n{title}\n")
        print(str(formatted_df))
        
        return formatted_df

# Usage
directory = '/content/drive/MyDrive/eval_outputs/'
results_table = process_evaluation_files(directory)