import pandas as pd
import glob, os, numpy as np
from collections import defaultdict
import re, krippendorff
from scipy.stats import pearsonr

def get_score_column(df, model):
    pref = {'claude-3-5-sonnet-20241022': 'anthropic/', 'gemini-1.5-pro-002': 'google/', 'gpt-4o-2024-08-06': 'openai/'}
    if model in pref:
        c = f"{pref[model]}{model}_score"
        if c in df.columns: return c
    b = f"{model}_score"
    if b in df.columns: return b
    for col in df.columns:
        if 'score' in col.lower() and model in col: return col
    return None

def parse_mean_sd(strval):
    try:
        match = re.match(r'([\-\d\.]+)\s*\(([\-\d\.]+)\)', str(strval))
        return (float(match.group(1)), float(match.group(2))) if match else (float(strval), np.nan)
    except: return np.nan, np.nan

def compute_krippendorff_alpha(judge_scores):
    reliability_data = np.array(judge_scores)
    return krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='interval')

def calculate_judge_correlations(all_files):
    judges = ['claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'gpt-4o-2024-08-06']
    judge_scores = defaultdict(list)
    
    for model_files in all_files.values():
        for fpath in model_files.values():
            df = pd.read_csv(fpath)
            for i, row in df.iterrows():
                for j in judges:
                    col = get_score_column(df, j)
                    if col and pd.notna(row[col]):
                        judge_scores[j].append(row[col])
    
    correlations = []
    judge_pairs = [(judges[0], judges[1]), (judges[0], judges[2]), (judges[1], judges[2])]
    
    for j1, j2 in judge_pairs:
        scores1, scores2 = np.array(judge_scores[j1]), np.array(judge_scores[j2])
        min_len = min(len(scores1), len(scores2))
        corr, _ = pearsonr(scores1[:min_len], scores2[:min_len])
        correlations.append(corr)
    
    judge_arrays = []
    max_len = max(len(scores) for scores in judge_scores.values())
    for j in judges:
        scores = judge_scores[j]
        scores.extend([np.nan] * (max_len - len(scores)))
        judge_arrays.append(scores)
    
    return correlations, compute_krippendorff_alpha(judge_arrays)

def print_correlation_table(correlations, overall_alpha):
    print("\nJudge Correlations and Overall Agreement:")
    print("-" * 50)
    print(f"{'Judge Pair':<30} | {'Correlation':>10}")
    print("-" * 50)
    for pair, corr in [("claude vs gemini", correlations[0]), ("claude vs gpt4", correlations[1]), ("gemini vs gpt4", correlations[2])]:
        print(f"{pair:<30} | {corr:>10.3f}")
    print("-" * 50)
    print(f"Overall Krippendorff's Alpha: {overall_alpha:.3f}")
    print("-" * 50 + "\n")

def process_files(directory):
    all_files = defaultdict(lambda: defaultdict(str))
    for fpath in glob.glob(os.path.join(directory, 'results_*.csv')):
        base = os.path.basename(fpath).replace('results_','').replace('.csv','')
        if '_run' in base:
            *modelparts, run_id = base.split('_run')
            all_files['_run'.join(modelparts)][run_id] = fpath
        else: all_files[base]['1'] = fpath

    correlations, overall_alpha = calculate_judge_correlations(all_files)
    print_correlation_table(correlations, overall_alpha)

    judges = ['claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'gpt-4o-2024-08-06']
    results = {}
    all_models_judge_scores = defaultdict(list)

    for base_model, run_map in all_files.items():
        all_raw_scores, judge_data = [], defaultdict(list)
        all_adj_scores, all_sp_scores = [], []
        model_judge_scores = defaultdict(list)

        for run_id, fpath in run_map.items():
            df_main = pd.read_csv(fpath)
            jcols_present = [(j, col) for j in judges if (col := get_score_column(df_main, j))]

            for i, row in df_main.iterrows():
                row_scores = []
                for judge_name, colname in jcols_present:
                    val = row[colname]
                    if pd.notna(val):
                        judge_data[judge_name].append(val)
                        row_scores.append(val)
                        model_judge_scores[judge_name].append(val)
                        all_models_judge_scores[judge_name].append(val)
                if row_scores:
                    all_raw_scores.append(np.mean(row_scores))

            if base_model in judges:
                other_judges = [j for j in judges if j != base_model]
                other_dfs = {oj: pd.read_csv(all_files[oj][run_id]) if oj in all_files and run_id in all_files[oj] else None for oj in other_judges}

                for i, row in df_main.iterrows():
                    rec_vals = [row[col] for oj in other_judges if (col := get_score_column(df_main, oj)) and not pd.isna(row[col])]
                    giv_vals = [odf.loc[i, bc] for oj, odf in other_dfs.items() if odf is not None and (bc := get_score_column(odf, base_model)) is not None and i < len(odf) and pd.notna(odf.loc[i, bc])]

                    if rec_vals and giv_vals:
                        h = np.mean(giv_vals)
                        adj_i = (sum(rec_vals) + h) / (len(other_judges) + 1)
                        row_judge_vals = [row[col] for _, col in jcols_present if pd.notna(row[col])]
                        raw_i = np.mean(row_judge_vals) if row_judge_vals else np.nan
                        sp_i = (raw_i - adj_i) * len(judges) if not np.isnan(raw_i) else np.nan
                    else:
                        adj_i, sp_i = np.nan, np.nan

                    all_adj_scores.append(adj_i)
                    all_sp_scores.append(sp_i)

        def mean_sd_str(values):
            arr = np.array(values, dtype=float)[~np.isnan(np.array(values, dtype=float))]
            return f"{arr[0]:.3f} (NaN)" if len(arr) == 1 else "NaN" if len(arr) < 1 else f"{arr.mean():.3f} ({arr.std(ddof=1):.3f})"

        n_points = sum(len(pd.read_csv(fpath)) for run_id, fpath in run_map.items())
        outrow = {"completions": len(run_map), "n": n_points,
                 "Unadjusted Average Score": mean_sd_str(all_raw_scores)}

        for jg in judges:
            outrow[jg] = mean_sd_str(judge_data[jg])

        if base_model in judges:
            outrow.update({"Average Score": mean_sd_str(all_adj_scores),
                         "Self-Preference": mean_sd_str(all_sp_scores)})
        else:
            outrow.update({"Average Score": outrow["Unadjusted Average Score"],
                         "Self-Preference": "NaN"})

        judge_scores_array = [scores for j in judges if (scores := model_judge_scores[j])]
        outrow["Krippendorff's Alpha"] = f"{compute_krippendorff_alpha(judge_scores_array):.3f}" if len(judge_scores_array) > 1 else "NaN"
        results[base_model] = outrow

    df_res = pd.DataFrame(results).T

    def wavg_and_sd(col, only_j=False):
        big_list = []
        for idx, row in df_res.iterrows():
            if only_j and idx not in judges: continue
            mval, _ = parse_mean_sd(row.get(col, "NaN"))
            if not pd.isna(row.get("n", np.nan)) and not np.isnan(mval):
                big_list.extend([mval] * int(row["n"]))
        return "NaN" if not big_list else f"{np.mean(big_list):.3f} (NaN)"

    all_judges_array = [scores for j in judges if (scores := all_models_judge_scores[j])]
    w_row = {"completions": "", "n": df_res["n"].sum()}
    w_row.update({c: wavg_and_sd(c, c == "Self-Preference") for c in df_res.columns if c not in ["completions", "n", "Krippendorff's Alpha"]})
    w_row["Krippendorff's Alpha"] = f"{compute_krippendorff_alpha(all_judges_array):.3f}" if len(all_judges_array) > 1 else "NaN"
    
    df_res.loc["Weighted Average"] = w_row
    return df_res

def main(dir_):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    df_res = process_files(dir_)
    print("\nLLM-as-a-Judge scores with pooled runs. Standard Deviation in brackets.\n")
    print(df_res)

if __name__ == "__main__":
    main("/content/drive/MyDrive/eval_outputs/")
