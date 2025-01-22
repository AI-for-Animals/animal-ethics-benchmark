import pandas as pd
import glob
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import re

def get_model_color(name):
    colors = {'claude':'#4A90E2','gpt':'#45B764','gemini':'#E6A040','mistral':'#9B6B9E','deepseek':'#E57373'}
    return next((v for k,v in colors.items() if k in name.lower()), '#757575')

def get_score_column(df, model):
    pref = {
        'claude-3-5-sonnet-20241022': 'anthropic/',
        'gemini-1.5-pro-002': 'google/',
        'gpt-4o-2024-08-06': 'openai/'
    }
    if model in pref:
        c = f"{pref[model]}{model}_score"
        if c in df.columns: return c
    b = f"{model}_score"
    if b in df.columns: return b
    for col in df.columns:
        if 'score' in col.lower() and model in col:
            return col
    return None

def parse_mean_se(val):
    """Parse a string like '0.123 (0.111, 0.135)' → (0.123, standard_error). Otherwise return (nan,nan)."""
    if val=="NaN": return np.nan,np.nan
    match = re.match(r'([\-\d\.]+)\s*\(([\-\d\.]+),\s*([\-\d\.]+)\)', str(val))
    if match:
        mv,lv,uv = float(match.group(1)), float(match.group(2)), float(match.group(3))
        return mv, (uv - mv)/1.96
    try:
        return float(val), np.nan
    except:
        return np.nan, np.nan

def compute_stats_with_resampling(scores_per_run, K):
    """
    Compute mean and SE following paper's methodology.
    Args:
        scores_per_run: List of arrays, each containing scores for one run
        K: Number of runs (resamples)
    """
    if K == 1:
        scores = scores_per_run[0]
        mean = np.mean(scores)
        se = np.std(scores, ddof=1) / np.sqrt(len(scores))
    else:
        # Convert to numpy for easier manipulation
        scores = np.array(scores_per_run)  # Shape: (K, n_questions)
        
        # First average K samples for each question
        question_means = np.mean(scores, axis=0)
        
        # Compute overall mean and SE
        mean = np.mean(question_means)
        base_se = np.std(question_means, ddof=1) / np.sqrt(len(question_means))
        
        # Apply variance reduction factor from paper
        reduction_factor = np.sqrt((1 + 2/K)/3)
        se = base_se * reduction_factor
    
    return mean, se

def process_files(directory):
    d = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(directory,'results_*.csv')):
        base = os.path.basename(f).replace('results_','').replace('.csv','')
        if '_run' in base:
            *m, r = base.split('_run')
            model = '_run'.join(m)
            d[model][r].append(f)
        else:
            d[base]['1'].append(f)
    d = {m:dict(runs) for m,runs in d.items()}

    judges = ['claude-3-5-sonnet-20241022','gemini-1.5-pro-002','gpt-4o-2024-08-06']
    res, vis = {}, []

    for base_model, runs in d.items():
        raw_scores_per_judge = defaultdict(list)  # Store all scores per judge across runs
        raw_avg, adj_avg = [], []
        
        K = len(runs)
        first_df = pd.read_csv(next(iter(runs.values()))[0])
        n_questions = len(first_df)
        
        # Collect scores across runs
        for run, files in runs.items():
            df = pd.read_csv(files[0])
            for j in judges:
                col = get_score_column(df, j)
                if col:
                    raw_scores_per_judge[j].append(df[col].values)
            
            # Calculate raw and adjusted averages for this run
            jcols = [get_score_column(df,j) for j in judges if get_score_column(df,j)]
            raw_avg.append(np.mean([df[c].mean() for c in jcols]))
            
            if base_model in judges:
                other = [x for x in judges if x!=base_model]
                rec,giv=[],[]
                for oj in other:
                    c = get_score_column(df,oj)
                    if c: rec.append(df[c].mean())
                for oj in other:
                    if oj in d and run in d[oj]:
                        df2 = pd.read_csv(d[oj][run][0])
                        bc = get_score_column(df2, base_model)
                        if bc: giv.append(df2[bc].mean())
                adj = np.nan
                if rec and giv:
                    h = np.mean(giv)
                    adj = (sum(rec)+h)/(len(other)+1)
                adj_avg.append(adj)
            else:
                adj_avg.append(np.nan)

        # Compute overall stats with proper resampling
        perjudge = {}
        for j in judges:
            if j in raw_scores_per_judge:
                scores = raw_scores_per_judge[j]
                if scores:
                    mm, st = compute_stats_with_resampling(scores, K)
                    if not np.isnan(mm):
                        ci_j = 1.96 * st
                        perjudge[j] = f"{mm:.3f} ({mm - ci_j:.3f}, {mm + ci_j:.3f})"

        # Calculate unadjusted average with proper resampling
        all_scores = [np.mean([df[c].values for c in jcols], axis=0) 
                     for run, files in runs.items() 
                     for df in [pd.read_csv(files[0])]
                     if (jcols := [get_score_column(df,j) for j in judges if get_score_column(df,j)])]
        
        unmean, se = compute_stats_with_resampling(all_scores, K)
        ci = 1.96 * se

        res[base_model] = {
            "completions": K,
            "n": K * n_questions,
            **perjudge,
            "Unadjusted Average Score": f"{unmean:.3f} ({unmean - ci:.3f}, {unmean + ci:.3f})"
        }

        if base_model in judges:
            val = [x for x in adj_avg if not np.isnan(x)]
            if val:
                # Compute adjusted average with proper resampling
                am, se_a = compute_stats_with_resampling([np.array(val)], K)
                ci_a = 1.96 * se_a
                res[base_model]["Average Score"] = f"{am:.3f} ({am - ci_a:.3f}, {am + ci_a:.3f})"
                
                # Compute self-preference with proper resampling
                n_j = len(judges)
                diffs = []
                for i in range(K):
                    if not np.isnan(adj_avg[i]):
                        diffs.append((raw_avg[i] - adj_avg[i]) * n_j)
                if diffs:
                    sp_m, sp_s = compute_stats_with_resampling([np.array(diffs)], K)
                    sp_ci = 1.96 * sp_s
                    res[base_model]["Self-Preference"] = f"{sp_m:.3f} ({sp_m - sp_ci:.3f}, {sp_m + sp_ci:.3f})"
                else:
                    res[base_model]["Self-Preference"] = "NaN"
            else:
                res[base_model]["Average Score"] = res[base_model]["Unadjusted Average Score"]
                res[base_model]["Self-Preference"] = "NaN"
        else:
            res[base_model]["Average Score"] = res[base_model]["Unadjusted Average Score"]
            res[base_model]["Self-Preference"] = "NaN"

        # For plotting: parse the "Average Score" or fallback
        sc_str = res[base_model]["Average Score"] if base_model in judges else res[base_model]["Unadjusted Average Score"]
        mVal,seVal = parse_mean_se(sc_str)
        if not np.isnan(mVal) and not np.isnan(seVal):
            half = 1.96 * seVal
            vis.append({
                "model": base_model,
                "score": mVal,
                "ci_lower": mVal - half,
                "ci_upper": mVal + half,
                "color": get_model_color(base_model)
            })
        else:
            # fallback
            vis.append({
                "model": base_model,
                "score": unmean,
                "ci_lower": unmean - ci,
                "ci_upper": unmean + ci,
                "color": get_model_color(base_model)
            })

    df_res = pd.DataFrame(res).T

    def wavg_ci(col, only_j=False):
        pts = []
        for i, row in df_res.iterrows():
            if only_j and i not in judges: continue
            mv, se_ = parse_mean_se(row[col]) if col in row else (np.nan, np.nan)
            if not np.isnan(mv):
                pts.append((row["n"], mv, se_))
        if not pts: return "NaN"
        N = sum(x[0] for x in pts)
        wm = sum(x[0]*x[1] for x in pts)/N
        var_ = 0
        for nn, mv, sev in pts:
            if not np.isnan(sev):
                var_ += (nn**2)*(sev**2)
        wvar = var_/(N**2)
        wse = np.sqrt(wvar)
        ci95 = 1.96 * wse
        return f"{wm:.3f} ({wm - ci95:.3f}, {wm + ci95:.3f})"

    # Weighted average row
    w_row = {}
    for c in df_res.columns:
        if c == "completions": w_row[c] = ""
        elif c == "n": w_row[c] = df_res["n"].sum()
        elif c == "Self-Preference": w_row[c] = wavg_ci(c, True)
        else: w_row[c] = wavg_ci(c, False)
    df_res.loc["Weighted Average"] = w_row

    return df_res, vis

def sort_models(vis_data):
    """Sort models according to specified ordering."""
    judges = ['claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'gpt-4o-2024-08-06']
    
    # Split data into categories
    judge_data = []
    claude_data = []
    gemini_data = []
    gpt_data = []
    other_data = []
    
    for d in vis_data:
        model = d["model"]
        if model in judges:
            judge_data.append((judges.index(model), d))  # Keep judge order
        elif model.startswith("claude"):
            claude_data.append((model, d))
        elif model.startswith("gemini"):
            gemini_data.append((model, d))
        elif model.startswith("gpt"):
            gpt_data.append((model, d))
        else:
            other_data.append((model, d))
    
    # Sort each category
    sorted_data = (
        [d for _, d in sorted(judge_data, key=lambda x: x[0])] +  # Judges in specified order
        [d for _, d in sorted(claude_data, key=lambda x: x[0])] +  # Other claude alphabetically
        [d for _, d in sorted(gemini_data, key=lambda x: x[0])] +  # Other gemini alphabetically
        [d for _, d in sorted(gpt_data, key=lambda x: x[0])] +     # Other gpt alphabetically
        [d for _, d in sorted(other_data, key=lambda x: x[0])]     # Rest alphabetically
    )
    
    return sorted_data

def plot_results(vis_data, save_path):
    """Plot results with models in specified order."""
    sorted_data = sort_models(vis_data)
    
    plt.figure(figsize=(12,6))
    x = np.arange(len(sorted_data))
    sc = [d["score"] for d in sorted_data]
    col = [d["color"] for d in sorted_data]
    yerr = np.array([[d["score"]-d["ci_lower"] for d in sorted_data],
                     [d["ci_upper"]-d["score"] for d in sorted_data]])
    
    bars = plt.bar(x, sc, color=col)
    plt.errorbar(x, sc, yerr=yerr, fmt='none', ecolor='gray', capsize=5)
    
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x()+b.get_width()/2., h, f"{h:.3f}", ha='center', va='bottom')
    
    plt.xlabel("Model")
    plt.ylabel("Average Score")
    plt.xticks(x, [d["model"] for d in sorted_data], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_model_subset(vis_data, save_path, filter_func):
    """Plot a subset of models based on filter function."""
    subset_data = [d for d in vis_data if filter_func(d["model"])]
    if subset_data:  # Only plot if we have data
        sorted_data = sort_models(subset_data)
        
        plt.figure(figsize=(12,6))
        x = np.arange(len(sorted_data))
        sc = [d["score"] for d in sorted_data]
        col = [d["color"] for d in sorted_data]
        yerr = np.array([[d["score"]-d["ci_lower"] for d in sorted_data],
                        [d["ci_upper"]-d["score"] for d in sorted_data]])
        
        bars = plt.bar(x, sc, color=col)
        plt.errorbar(x, sc, yerr=yerr, fmt='none', ecolor='gray', capsize=5)
        
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x()+b.get_width()/2., h, f"{h:.3f}", ha='center', va='bottom')
        
        plt.xlabel("Model")
        plt.ylabel("Average Score")
        plt.xticks(x, [d["model"] for d in sorted_data], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def main(dir_, save_path="results_plot.png"):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    
    df, vis = process_files(dir_)
    print("\nLLM-as-a-Judge scores with pooled runs. 95% c.i. in brackets.\n")
    print(df)
    
    # Main plot with all models
    plot_results(vis, save_path)
    
    # Define judges list
    judges = ['claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'gpt-4o-2024-08-06']
    
    # Plot judges only
    plot_model_subset(vis, "proprietary_large.png", 
                     lambda m: m in judges)
    
    # Plot other proprietary models
    plot_model_subset(vis, "proprietary_compact.png", 
                     lambda m: m not in judges and 
                     (m.startswith("claude") or 
                      m.startswith("gemini") or 
                      m.startswith("gpt")))
    
    # Plot open models
    plot_model_subset(vis, "open.png", 
                     lambda m: m not in judges and 
                     not (m.startswith("claude") or 
                          m.startswith("gemini") or 
                          m.startswith("gpt")))

if __name__ == "__main__":
    main("/content/drive/MyDrive/eval_outputs/")
