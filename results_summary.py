import pandas as pd, glob, os, numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import re

def get_model_color(name):
    colors = {'claude':'#4A90E2','gpt':'#45B764','gemini':'#E6A040','mistral':'#9B6B9E','deepseek':'#E57373'}
    return next((v for k,v in colors.items() if k in name.lower()), '#757575')

def get_score_column(df, model):
    pref = {'claude-3-5-sonnet-20241022': 'anthropic/','gemini-1.5-pro-002': 'google/','gpt-4o-2024-08-06': 'openai/'}
    if model in pref and (c := f"{pref[model]}{model}_score") in df.columns: return c
    if (b := f"{model}_score") in df.columns: return b
    return next((col for col in df.columns if 'score' in col.lower() and model in col), None)

def parse_mean_se(val):
    if val=="NaN": return np.nan,np.nan
    if match := re.match(r'([\-\d\.]+)\s*\(([\-\d\.]+),\s*([\-\d\.]+)\)', str(val)):
        mv,lv,uv = map(float, match.groups())
        return mv, (uv - mv)/1.96
    try: return float(val), np.nan
    except: return np.nan, np.nan

def compute_stats_with_resampling(scores_per_run, K):
    if K == 1:
        scores = scores_per_run[0]
        return np.mean(scores), np.std(scores, ddof=1) / np.sqrt(len(scores))
    scores = np.array(scores_per_run)
    question_means = np.mean(scores, axis=0)
    mean, base_se = np.mean(question_means), np.std(question_means, ddof=1) / np.sqrt(len(question_means))
    return mean, base_se * np.sqrt((1 + 2/K)/3)

def sort_models_df(df):
    judges = ['claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'gpt-4o-2024-08-06']
    categories = defaultdict(list)
    for idx in df.index:
        if idx == "Weighted Average": continue
        if idx in judges: categories['judge'].append((judges.index(idx), idx))
        elif any(idx.startswith(p) for p in ['claude', 'gemini', 'gpt']):
            categories[idx.split('-')[0]].append((idx, idx))
        else: categories['other'].append((idx, idx))
    sorted_indices = [idx for _, idx in sorted(categories['judge'])]
    for k in ['claude', 'gemini', 'gpt', 'other']:
        sorted_indices.extend(idx for _, idx in sorted(categories[k]))
    return df.reindex(sorted_indices + ["Weighted Average"])

def process_files(directory):
    d = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(directory,'results_*.csv')):
        base = os.path.basename(f).replace('results_','').replace('.csv','')
        *m, r = base.split('_run') if '_run' in base else (base, '1')
        d['_run'.join(m)][r].append(f)
    d = {m:dict(runs) for m,runs in d.items()}
    
    judges = ['claude-3-5-sonnet-20241022','gemini-1.5-pro-002','gpt-4o-2024-08-06']
    res, vis = {}, []

    for base_model, runs in d.items():
        raw_scores_per_judge = defaultdict(list)
        raw_avg, adj_avg = [], []
        K = len(runs)
        first_df = pd.read_csv(next(iter(runs.values()))[0])
        n_questions = len(first_df)
        
        for run, files in runs.items():
            df = pd.read_csv(files[0])
            for j in judges:
                if col := get_score_column(df, j):
                    raw_scores_per_judge[j].append(df[col].values)
            
            jcols = [get_score_column(df,j) for j in judges if get_score_column(df,j)]
            raw_avg.append(np.mean([df[c].mean() for c in jcols]))
            
            if base_model in judges:
                other = [x for x in judges if x!=base_model]
                rec = [df[c].mean() for oj in other if (c := get_score_column(df,oj))]
                giv = []
                for oj in other:
                    if oj in d and run in d[oj]:
                        df2 = pd.read_csv(d[oj][run][0])
                        if bc := get_score_column(df2, base_model):
                            giv.append(df2[bc].mean())
                adj_avg.append((sum(rec) + np.mean(giv))/(len(other)+1) if rec and giv else np.nan)
            else:
                adj_avg.append(np.nan)

        perjudge = {j: f"{mm:.3f} ({mm - 1.96*st:.3f}, {mm + 1.96*st:.3f})"
                    for j in judges if j in raw_scores_per_judge 
                    and (scores := raw_scores_per_judge[j])
                    and not np.isnan(mm := compute_stats_with_resampling(scores, K)[0])
                    and not np.isnan(st := compute_stats_with_resampling(scores, K)[1])}

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

        if base_model in judges and (val := [x for x in adj_avg if not np.isnan(x)]):
            am, se_a = compute_stats_with_resampling([np.array(val)], K)
            ci_a = 1.96 * se_a
            res[base_model]["Average Score"] = f"{am:.3f} ({am - ci_a:.3f}, {am + ci_a:.3f})"
            
            diffs = [(raw_avg[i] - adj_avg[i]) * len(judges) for i in range(K) if not np.isnan(adj_avg[i])]
            if diffs:
                sp_m, sp_s = compute_stats_with_resampling([np.array(diffs)], K)
                sp_ci = 1.96 * sp_s
                res[base_model]["Self-Preference"] = f"{sp_m:.3f} ({sp_m - sp_ci:.3f}, {sp_m + sp_ci:.3f})"
            else: res[base_model]["Self-Preference"] = "NaN"
        else:
            res[base_model].update({"Average Score": res[base_model]["Unadjusted Average Score"], 
                                  "Self-Preference": "NaN"})

        sc_str = res[base_model]["Average Score" if base_model in judges else "Unadjusted Average Score"]
        if not np.isnan((mVal := parse_mean_se(sc_str)[0])) and not np.isnan((seVal := parse_mean_se(sc_str)[1])):
            half = 1.96 * seVal
            vis.append({"model": base_model, "score": mVal, "ci_lower": mVal - half,
                       "ci_upper": mVal + half, "color": get_model_color(base_model)})
        else:
            vis.append({"model": base_model, "score": unmean, "ci_lower": unmean - ci,
                       "ci_upper": unmean + ci, "color": get_model_color(base_model)})

    df_res = pd.DataFrame(res).T

    def wavg_ci(col, only_j=False):
        pts = [(row["n"], mv, se_) for i, row in df_res.iterrows()
               if (not only_j or i in judges)
               and not np.isnan(mv := parse_mean_se(row.get(col, "NaN"))[0])
               and not np.isnan(se_ := parse_mean_se(row.get(col, "NaN"))[1])]
        if not pts: return "NaN"
        N = sum(x[0] for x in pts)
        wm = sum(x[0]*x[1] for x in pts)/N
        wse = np.sqrt(sum((nn**2)*(sev**2) for nn,_,sev in pts)/(N**2))
        return f"{wm:.3f} ({wm - 1.96*wse:.3f}, {wm + 1.96*wse:.3f})"

    df_res.loc["Weighted Average"] = {
        "completions": "",
        "n": df_res["n"].sum(),
        **{c: wavg_ci(c, c=="Self-Preference") for c in df_res.columns if c not in ["completions","n"]}
    }
    return sort_models_df(df_res), vis

def sort_models(vis_data):
    judges = ['claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'gpt-4o-2024-08-06']
    categories = defaultdict(list)
    for d in vis_data:
        model = d["model"]
        if model in judges: categories['judge'].append((judges.index(model), d))
        elif model.startswith(('claude','gemini','gpt')):
            categories[model.split('-')[0]].append((model, d))
        else: categories['other'].append((model, d))
    return ([d for _, d in sorted(categories['judge'])] +
            [d for _, d in sorted(categories['claude'])] +
            [d for _, d in sorted(categories['gemini'])] +
            [d for _, d in sorted(categories['gpt'])] +
            [d for _, d in sorted(categories['other'])])

def plot_results(vis_data, save_path):
    plt.figure(figsize=(14,8))
    plt.rcParams.update({'font.size': 14})
    sorted_data = sort_models(vis_data)
    x = np.arange(len(sorted_data))
    plt.bar(x, [d["score"] for d in sorted_data], color=[d["color"] for d in sorted_data])
    plt.errorbar(x, [d["score"] for d in sorted_data],
                yerr=[[d["score"]-d["ci_lower"] for d in sorted_data],
                      [d["ci_upper"]-d["score"] for d in sorted_data]], 
                fmt='none', ecolor='gray', capsize=5)
    plt.grid(False)
    plt.xticks(x, [d["model"] for d in sorted_data], rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=1.2)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_model_subset(vis_data, save_path, filter_func):
    if subset_data := [d for d in vis_data if filter_func(d["model"])]:
        plt.figure(figsize=(14,8))
        plt.rcParams.update({'font.size': 14})
        sorted_data = sort_models(subset_data)
        x = np.arange(len(sorted_data))
        plt.bar(x, [d["score"] for d in sorted_data], color=[d["color"] for d in sorted_data])
        plt.errorbar(x, [d["score"] for d in sorted_data],
                    yerr=[[d["score"]-d["ci_lower"] for d in sorted_data],
                          [d["ci_upper"]-d["score"] for d in sorted_data]],
                    fmt='none', ecolor='gray', capsize=5)
        plt.grid(False)
        plt.xticks(x, [d["model"] for d in sorted_data], rotation=0, fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout(pad=1.2)
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
    
    # Save plots and print paths
    save_paths = {
        'main': os.path.abspath(save_path),
        'judges': os.path.abspath("proprietary_large.png"),
        'proprietary': os.path.abspath("proprietary_compact.png"),
        'open': os.path.abspath("open.png")
    }
    
    plot_results(vis, save_paths['main'])
    judges = ['claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'gpt-4o-2024-08-06']
    
    plot_model_subset(vis, save_paths['judges'], lambda m: m in judges)
    plot_model_subset(vis, save_paths['proprietary'], 
                     lambda m: m not in judges and m.startswith(('claude','gemini','gpt')))
    plot_model_subset(vis, save_paths['open'],
                     lambda m: m not in judges and not m.startswith(('claude','gemini','gpt')))
    
    print("\nPlots saved at:")
    for plot_type, path in save_paths.items():
        print(f"{plot_type}: {path}")
        
if __name__ == "__main__":
    main("/content/drive/MyDrive/eval_outputs/")
