import pandas as pd, glob, os, numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import re

def get_model_color(name):
    colors = {'claude':'#4A90E2','gpt':'#45B764','gemini':'#E6A040','mistral':'#9B6B9E','deepseek':'#E57373'}
    return next((v for k,v in colors.items() if k in name.lower()), '#757575')

def get_score_column(df, model):
    qualified = {
        'claude-3-5-sonnet-20241022': 'anthropic/claude-3-5-sonnet-20241022_score',
        'gemini-1.5-pro-002': 'google/gemini-1.5-pro-002_score',
        'gpt-4o-2024-08-06': 'openai/gpt-4o-2024-08-06_score'
    }
    return qualified.get(model) or next((col for col in df.columns if 'score' in col.lower() and model in col), None)

def get_related_judge(model):
    relations = {
        'claude-3-5-haiku-20241022': 'claude-3-5-sonnet-20241022',
        'gemini-1.5-flash-002': 'gemini-1.5-pro-002',
        'gpt-4o-mini-2024-07-18': 'gpt-4o-2024-08-06'
    }
    return relations.get(model)

def compute_stats(scores, K=1):
    if isinstance(scores[0], (list, np.ndarray)):
        scores = np.array(scores)
        means = np.mean(scores, axis=0)
        n_questions = len(means)
        
        mean_val = np.mean(means)
        if n_questions < 2:
            se_val = 0.0
        else:
            se_val = np.std(means, ddof=1) / np.sqrt(n_questions)
            if K > 1:
                se_val *= np.sqrt((1 + 2/K)/3)
        return mean_val, se_val
    
    n_points = len(scores)
    return np.mean(scores), 0.0 if n_points < 2 else np.std(scores, ddof=1) / np.sqrt(n_points)

def format_stats(mean, se):
    ci = 1.96 * se
    return f"{mean:.3f} ({mean - ci:.3f}, {mean + ci:.3f})"

def sort_by_category(items, judges, key_func=lambda x: x):
    categories = defaultdict(list)
    for item in items:
        model = key_func(item)
        if model in judges:
            categories['judge'].append((judges.index(model), item))
        elif model.startswith(('claude','gemini','gpt')):
            categories[model.split('-')[0]].append((model, item))
        else:
            categories['other'].append((model, item))
    
    result = [item for _, item in sorted(categories['judge'])]
    for k in ['claude', 'gemini', 'gpt', 'other']:
        result.extend(item for _, item in sorted(categories[k]))
    return result

def process_files(directory):
    d = {}
    for f in glob.glob(os.path.join(directory,'results_*.csv')):
        base = os.path.basename(f).replace('results_','').replace('.csv','')
        model, run = base.split('_run') if '_run' in base else (base, '1')
        if model not in d:
            d[model] = {}
        d[model][run] = f
    
    judges = ['claude-3-5-sonnet-20241022','gemini-1.5-pro-002','gpt-4o-2024-08-06']
    res = {}

    def get_adjusted_scores(df, model, judge_scores=None):
        if model in judges:
            other = [x for x in judges if x != model]
            rec = [df[get_score_column(df, oj)].values for oj in other if get_score_column(df, oj)]
            giv = judge_scores if judge_scores is not None else []
            return (np.sum(rec, axis=0) + giv) / len(judges) if rec and len(giv) else None
        
        if related_judge := get_related_judge(model):
            other = [x for x in judges if x != related_judge]
            rec = [df[get_score_column(df, oj)].values for oj in other if get_score_column(df, oj)]
            related_scores = []
            for oj in judges:
                if oj != related_judge:
                    related_model = next((m for m in d if get_related_judge(m) == oj), None)
                    if related_model and related_model in d:
                        df2 = pd.read_csv(next(iter(d[related_model].values())))
                        if bc := get_score_column(df2, related_judge):
                            related_scores.append(df2[bc].values)
            return (np.sum(rec, axis=0) + np.mean(related_scores, axis=0)) / len(judges) if rec and related_scores else None
        return None

    for base_model, runs in d.items():
        K = len(runs)
        df_first = pd.read_csv(next(iter(runs.values())))
        n_questions = len(df_first)

        all_scores = []
        for fpath_ in runs.values():
            df_ = pd.read_csv(fpath_)
            jcols = [get_score_column(df_, j) for j in judges if get_score_column(df_, j)]
            if jcols:
                all_scores.append(np.mean([df_[c].values for c in jcols], axis=0))
        mean, se = compute_stats(all_scores, K)
        adj_mean, adj_se = mean, se
        diff_mean, diff_se = np.nan, np.nan

        if base_model in judges or get_related_judge(base_model):
            if K > 1:
                adj_scores_per_run = []
                raw_scores_per_run = []

                for run_id, fpath in runs.items():
                    df_ = pd.read_csv(fpath)
                    jcols = [get_score_column(df_, j) for j in judges if get_score_column(df_, j)]
                    raw_this_run = np.mean([df_[c].values for c in jcols], axis=0)
                    raw_scores_per_run.append(raw_this_run)

                    if base_model in judges:
                        other_judges = [x for x in judges if x != base_model]
                        judge_scores_arrays = []
                        for oj in other_judges:
                            if oj in d and run_id in d[oj]:
                                df2 = pd.read_csv(d[oj][run_id])
                                if bc := get_score_column(df2, base_model):
                                    judge_scores_arrays.append(df2[bc].values)
                        combined_judge_scores = np.mean(judge_scores_arrays, axis=0) if judge_scores_arrays else None
                        adj = get_adjusted_scores(df_, base_model, judge_scores=combined_judge_scores)
                    else:
                        adj = get_adjusted_scores(df_, base_model)
                    
                    if adj is not None:
                        adj_scores_per_run.append(adj)

                if adj_scores_per_run and len(adj_scores_per_run) == K:
                    adj_mean, adj_se = compute_stats(adj_scores_per_run, K)
                    diffs = [(raw_scores_per_run[i] - adj_scores_per_run[i]) * len(judges) for i in range(K)]
                    diff_mean, diff_se = compute_stats([np.mean(d) for d in diffs], K)

            else:
                fpath_single = next(iter(runs.values()))
                df_single = pd.read_csv(fpath_single)
                
                if base_model in judges:
                    other_judges = [x for x in judges if x != base_model]
                    judge_scores_arrays = []
                    for oj in other_judges:
                        if oj in d and '1' in d[oj]:
                            if fpath_single_oj := d[oj].get('1'):
                                df2 = pd.read_csv(fpath_single_oj)
                                if bc := get_score_column(df2, base_model):
                                    judge_scores_arrays.append(df2[bc].values)
                    combined_judge_scores = np.mean(judge_scores_arrays, axis=0) if judge_scores_arrays else None
                    adj_scores = get_adjusted_scores(df_single, base_model, judge_scores=combined_judge_scores)
                else:
                    adj_scores = get_adjusted_scores(df_single, base_model)

                if adj_scores is not None:
                    adj_mean, adj_se = compute_stats([adj_scores])
                    diffs = (all_scores[0] - adj_scores) * len(judges)
                    diff_mean, diff_se = compute_stats([diffs])

        row_dict = {"completions": K, "n": K * n_questions}

        for j in judges:
            col_j = get_score_column(df_first, j)
            if col_j:
                judge_run_scores = []
                for fpath_ in runs.values():
                    df_tmp = pd.read_csv(fpath_)
                    if ctmp := get_score_column(df_tmp, j):
                        judge_run_scores.append(df_tmp[ctmp].values)
                if judge_run_scores:
                    jm, js = compute_stats(judge_run_scores, K)
                    row_dict[j] = format_stats(jm, js)

        row_dict["Unadjusted Average Score"] = format_stats(mean, se)
        row_dict["Average Score"] = format_stats(adj_mean, adj_se)
        row_dict["Company Preference" if get_related_judge(base_model) else "Self-Preference" if base_model in judges else "Company Preference"] = format_stats(diff_mean, diff_se) if not np.isnan(diff_mean) else "NaN"
        res[base_model] = row_dict

    df_res = pd.DataFrame(res).T
    df_res = df_res.reindex(sort_by_category(df_res.index, judges))

    def parse_stats_str(stat_str):
        if not isinstance(stat_str, str) or stat_str == "NaN":
            return None, None
        nums = re.findall(r"(-?\d+\.\d+)", stat_str)
        return (float(nums[0]), (float(nums[2]) - float(nums[0]))/1.96) if len(nums) == 3 else (None, None)

    big_n = df_res["n"].sum()
    weighted_avgs = {}
    for col in df_res.columns:
        if col in ["completions", "n"]:
            continue
        points = []
        for idx, row_ in df_res.iterrows():
            if idx != "Weighted Average":
                m_, se_ = parse_stats_str(row_.get(col, "NaN"))
                if m_ is not None and se_ is not None:
                    points.append((row_["n"], m_, se_))
        if points:
            wmean = sum(n_ * m_ for n_, m_, _ in points) / big_n
            wse = np.sqrt(sum((n_**2)*(sev**2) for n_, _, sev in points)) / big_n
            weighted_avgs[col] = format_stats(wmean, wse)
        else:
            weighted_avgs[col] = "NaN"

    df_res.loc["Weighted Average"] = {"completions": "", "n": big_n, **weighted_avgs}
    return df_res

def plot_models_from_df(df_res, save_path, subset_func=None, rotation=45, title=""):
    data_for_plot = []
    judges = ['claude-3-5-sonnet-20241022','gemini-1.5-pro-002','gpt-4o-2024-08-06']

    def parse_stats_str(stat_str):
        if not isinstance(stat_str, str) or stat_str == "NaN":
            return None, None, None
        nums = re.findall(r"(-?\d+\.\d+)", stat_str)
        return (float(nums[0]), float(nums[1]), float(nums[2])) if len(nums) == 3 else (None, None, None)

    for idx, row in df_res.iterrows():
        if idx != "Weighted Average" and (not subset_func or subset_func(idx)):
            mean_, low_, high_ = parse_stats_str(row.get("Average Score", "NaN"))
            if mean_ is not None:
                data_for_plot.append({
                    "model": idx,
                    "score": mean_,
                    "ci_lower": low_,
                    "ci_upper": high_,
                    "color": get_model_color(idx)
                })

    sorted_data = sort_by_category(data_for_plot, judges, key_func=lambda x: x["model"])
    plt.figure(figsize=(14,8))
    plt.rcParams.update({'font.size': 14})
    x = np.arange(len(sorted_data))
    plt.bar(x, [d["score"] for d in sorted_data], color=[d["color"] for d in sorted_data])
    plt.errorbar(x, [d["score"] for d in sorted_data],
                yerr=[[d["score"] - d["ci_lower"] for d in sorted_data],
                      [d["ci_upper"] - d["score"] for d in sorted_data]],
                fmt='none', ecolor='gray', capsize=5)
    plt.grid(False)
    plt.xticks(x, [d["model"] for d in sorted_data], rotation=rotation,
               ha='right' if rotation == 45 else 'center')
    if title:
        plt.title(title)
    plt.tight_layout(pad=1.2)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main(dir_, save_path="results_plot.png"):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)

    df_res = process_files(dir_)
    print("\nLLM-as-a-Judge scores with pooled runs. 95% c.i. in brackets.\n")
    print(df_res)

    main_path = os.path.abspath(save_path)
    plot_models_from_df(df_res, main_path, subset_func=None, rotation=45, title="All Models")
    print(f"main: {main_path}")

    judges = ['claude-3-5-sonnet-20241022','gemini-1.5-pro-002','gpt-4o-2024-08-06']
    path_judges = os.path.abspath("proprietary_large.png")
    plot_models_from_df(df_res, path_judges, subset_func=lambda m: m in judges,
                       rotation=0, title="Judges Only")
    print(f"judges: {path_judges}")

    path_prop = os.path.abspath("proprietary_compact.png")
    plot_models_from_df(df_res, path_prop,
                       subset_func=lambda m: (m not in judges and
                                           m.startswith(('claude','gemini','gpt'))),
                       rotation=0, title="Proprietary Non-Judge Models")
    print(f"proprietary: {path_prop}")

    path_open = os.path.abspath("open.png")
    plot_models_from_df(df_res, path_open,
                       subset_func=lambda m: (m not in judges and
                                           not m.startswith(('claude','gemini','gpt'))),
                       rotation=0, title="Open Models")
    print(f"open: {path_open}")

if __name__ == "__main__":
    main("/content/drive/MyDrive/eval_outputs/")
