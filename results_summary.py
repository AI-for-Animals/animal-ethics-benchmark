import pandas as pd, glob, os, numpy as np, re, argparse, krippendorff
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, rankdata

def detect_judges(directory):
    files = glob.glob(os.path.join(directory, 'results_*.csv'))
    for f in files:
        try:
            df = pd.read_csv(f)
        except:
            continue
        pat = re.compile(r"^(?:.*/)?([^/]+)/([^/]+)_assessment\s*$", re.IGNORECASE)
        judges = {}
        for col in df.columns:
            c = col.strip()
            if c.endswith("_assessment"):
                m = pat.match(c)
                if m:
                    judges[m.group(2)] = m.group(1).lower()
        if judges:
            return list(judges.keys()), judges
    return [], {}

def get_score_column(df, model):
    pat = re.compile(rf".*{re.escape(model)}_score\s*$", re.IGNORECASE)
    for col in df.columns:
        if pat.match(col):
            return col
    for col in df.columns:
        if (("score" in col.lower() or "assessment" in col.lower()) and model.lower() in col.lower()):
            return col
    return None

def mean_se_ci_str(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if not len(arr): return "NaN"
    if len(arr)==1: return f"{arr[0]:.3f} (NaN) [NaN, NaN]"
    m = arr.mean(); se = arr.std(ddof=1)/np.sqrt(len(arr))
    return f"{m:.3f} ({se:.3f}) [{m-1.96*se:.3f}, {m+1.96*se:.3f}]"

def compute_krippendorff_alpha(scores):
    arr = np.array(scores)
    uniq = np.unique(arr[~np.isnan(arr)])
    if uniq.size <= 1: return np.nan
    return krippendorff.alpha(reliability_data=arr, level_of_measurement='interval')

def calculate_judge_correlations(all_files, judges):
    jscores = defaultdict(list)
    for files in all_files.values():
        for f in files.values():
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                for j in judges:
                    col = get_score_column(df, j)
                    if col and pd.notna(row[col]):
                        try: jscores[j].append(float(row[col]))
                        except: continue
    counts = {j: len(jscores[j]) for j in judges}
    pair_results = []
    for i in range(len(judges)):
        for j in range(i+1, len(judges)):
            j1, j2 = judges[i], judges[j]
            s1, s2 = np.array(jscores[j1]), np.array(jscores[j2])
            n_obs = min(len(s1), len(s2)) if len(s1) and len(s2) else 0
            corr = pearsonr(s1[:n_obs], s2[:n_obs])[0] if n_obs else np.nan
            pair_results.append((f"{j1} vs {j2}", n_obs, corr))
    jarrays = []
    if jscores:
        max_len = max(len(v) for v in jscores.values())
        for j in judges:
            arr = jscores[j][:]
            arr.extend([np.nan]*(max_len-len(arr)))
            jarrays.append(arr)
        overall_alpha = compute_krippendorff_alpha(jarrays)
    else:
        overall_alpha = np.nan
    return pair_results, overall_alpha, counts

def infer_provider(model, df):
    for col in df.columns:
        if model.lower() in col.lower() and "/" in col:
            return col.split("/")[0].lower()
    return None

def get_related_judge_for_model(model, df, judges, judge_providers):
    if model.lower() in [j.lower() for j in judges]:
        return model
    prov = infer_provider(model, df)
    if prov and prov in judge_providers.values():
        for j, p in judge_providers.items():
            if p == prov:
                return j
    return None

def compute_global_judge_harshness(all_files, judges, judge_providers):
    scores = {j: [] for j in judges}
    for model, files in all_files.items():
        try:
            df_first = pd.read_csv(next(iter(files.values())))
        except:
            continue
        prov = infer_provider(model, df_first)
        for f in files.values():
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                for j in judges:
                    col = get_score_column(df, j)
                    if col and pd.notna(row[col]):
                        try: val = float(row[col])
                        except: continue
                        if prov is not None and prov != judge_providers[j]:
                            scores[j].append(val)
    num, fmt = {}, {}
    for j in judges:
        if scores[j]:
            avg = np.mean(scores[j])
            num[j] = avg; fmt[j] = mean_se_ci_str(scores[j])
        else:
            num[j], fmt[j] = np.nan, "NaN"
    return num, fmt

def parse_mean(s):
    try: 
        return float(s.split()[0])
    except:
        return np.nan

def process_files(directory):
    all_files = defaultdict(lambda: defaultdict(str))
    for f in glob.glob(os.path.join(directory, 'results_*.csv')):
        base = os.path.basename(f).replace('results_', '').replace('.csv', '')
        if '_run' in base:
            *parts, run = base.split('_run')
            key = '_run'.join(parts)
            all_files[key][run] = f
        else:
            all_files[base]['1'] = f
    judges, judge_providers = detect_judges(directory)
    global_harshness_num, global_harshness_fmt = compute_global_judge_harshness(all_files, judges, judge_providers)
    pair_results, overall_corr_alpha, _ = calculate_judge_correlations(all_files, judges)
    results = {}
    for model, files in all_files.items():
        try: df_first = pd.read_csv(next(iter(files.values())))
        except: continue
        related_judge = get_related_judge_for_model(model, df_first, judges, judge_providers)
        is_judge = (model.lower() in [j.lower() for j in judges])
        role = "JUDGE" if is_judge else ("Related" if related_judge is not None else "independent")
        raw_all, adj_all, sp_all = [], [], []
        judge_data = {j: [] for j in judges}
        giv_all = []
        total = 0
        for run, f in files.items():
            try: df = pd.read_csv(f)
            except: continue
            for _, row in df.iterrows():
                rjs = {}
                for j in judges:
                    col = get_score_column(df, j)
                    if col and pd.notna(row[col]):
                        try: val = float(row[col])
                        except: continue
                        rjs[j] = val
                        judge_data[j].append(val)
                raw_val = np.mean(list(rjs.values())) if rjs else np.nan
                if related_judge is not None:
                    if is_judge:
                        other = [oj for oj in judges if oj.lower() != model.lower()]
                        rec = [rjs[oj] for oj in other if oj in rjs]
                        giv = []
                        for oj in other:
                            if oj in all_files and run in all_files[oj]:
                                try:
                                    df2 = pd.read_csv(all_files[oj][run][0])
                                    col2 = get_score_column(df2, model)
                                    if col2:
                                        arr = df2[col2].dropna().astype(float).values
                                        if len(arr): giv.append(np.mean(arr))
                                except: continue
                        if not giv:
                            fb = global_harshness_num.get(model, np.nan)
                            if not np.isnan(fb): giv = [fb]
                        adj_val = (sum(rec) + np.mean(giv)) / 3.0 if (rec and giv) else ((sum(rec) + fb) / 3.0 if rec and not np.isnan(fb) else np.nan)
                        if giv: giv_all.extend(giv)
                    else:
                        other = [oj for oj in judges if oj.lower() != related_judge.lower()]
                        rec = [rjs[oj] for oj in other if oj in rjs]
                        H = global_harshness_num.get(related_judge, np.nan)
                        adj_val = (sum(rec) + H) / 3.0 if (rec and not np.isnan(H)) else np.nan
                else:
                    adj_val = raw_val
                sp_val = (raw_val - adj_val) * 3 if (adj_val is not None and not np.isnan(adj_val)) else np.nan
                raw_all.append(raw_val); adj_all.append(adj_val); sp_all.append(sp_val); total += 1
        comp = len(files); n_points = total
        outrow = {"Role": role, "completions": comp, "n": n_points, "Unadjusted Average Score": mean_se_ci_str(raw_all)}
        for j in judges:
            outrow[j] = mean_se_ci_str(judge_data[j])
        if related_judge is not None:
            outrow["Average Score"] = mean_se_ci_str(adj_all)
            outrow["Self-Preference"] = mean_se_ci_str(sp_all)
            if is_judge:
                outrow["Related Judge Harshness"] = mean_se_ci_str(giv_all) if giv_all else "n/a"
            else:
                outrow["Related Judge Harshness"] = global_harshness_fmt.get(related_judge, "n/a")
        else:
            outrow["Average Score"] = outrow["Unadjusted Average Score"]
            outrow["Self-Preference"] = "n/a"
            outrow["Related Judge Harshness"] = "n/a"
        results[model] = outrow
    df_res = pd.DataFrame(results).T
    order = ["Role", "completions", "n", "Average Score", "Unadjusted Average Score"] + judges + ["Self-Preference", "Related Judge Harshness"]
    order = [c for c in order if c in df_res.columns]
    df_res = df_res[order]
    df_res["AvgNumeric"] = df_res["Average Score"].apply(parse_mean)
    df_res = df_res.sort_values(by="AvgNumeric", ascending=False).drop(columns=["AvgNumeric"])
    def wavg(col, df):
        pts = []
        for _, row in df.iterrows():
            try: n_val = int(row["n"])
            except: continue
            m_val = parse_mean(row.get(col, "NaN"))
            if not np.isnan(m_val): pts.extend([m_val]*n_val)
        if not pts: return "NaN"
        arr = np.array(pts, dtype=float)
        m = arr.mean(); se = arr.std(ddof=1)/np.sqrt(len(arr))
        return f"{m:.3f} ({se:.3f}) [{m-1.96*se:.3f}, {m+1.96*se:.3f}]"
    w_row = {"Role": "", "completions": "", "n": df_res["n"].sum()}
    for col in df_res.columns:
        if col not in ["Role", "completions", "n"]:
            w_row[col] = wavg(col, df_res)
    df_res.loc["Weighted Average"] = w_row
    pair_results, overall_corr_alpha, _ = calculate_judge_correlations(all_files, judges)
    corr_table = []
    for (pair, n_obs, corr) in pair_results:
        j1, j2 = pair.split(" vs ")
        x = df_res.loc[df_res.index!="Weighted Average", j1].apply(parse_mean).values
        y = df_res.loc[df_res.index!="Weighted Average", j2].apply(parse_mean).values
        rank_corr = spearmanr(x, y)[0] if len(x)>0 and len(y)>0 else np.nan
        corr_table.append({"Judge Pair": pair, "n": n_obs, "Correlation": corr, "Ranking Correlation": rank_corr})
    corr_df = pd.DataFrame(corr_table)
    ranking_matrix = []
    models_rank = df_res.index[df_res.index!="Weighted Average"]
    for j in judges:
        scores = df_res.loc[models_rank, j].apply(parse_mean).values
        ranks = rankdata(scores) if len(scores)>0 else np.array([])
        ranking_matrix.append(ranks)
    overall_rank_alpha = compute_krippendorff_alpha(ranking_matrix) if ranking_matrix and ranking_matrix[0].size>0 else np.nan
    return df_res, corr_df, overall_corr_alpha, overall_rank_alpha

def main(input_directory):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.expand_frame_repr", True)
    df_res, corr_df, overall_corr_alpha, overall_rank_alpha = process_files(input_directory)
    print("\nLLM-as-a-Judge scores with pooled runs.\n")
    print(df_res)
    print("\nJudge Correlations and Ranking Correlations:")
    print(corr_df.to_string(index=False, float_format="%.3f"))
    print(f"\nOverall Krippendorff's Alpha (Correlation): {overall_corr_alpha:.3f}")
    print(f"Overall Krippendorff's Alpha (Ranking):   {overall_rank_alpha:.3f}\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", default="/content/aha/results")
    args = parser.parse_args()
    main(args.input_directory)
