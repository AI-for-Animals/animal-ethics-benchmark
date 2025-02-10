import pandas as pd
import glob, os, numpy as np
from collections import defaultdict
import re, krippendorff
from scipy.stats import pearsonr, spearmanr, rankdata

# -------------------------------
# Global Settings and Mappings
# -------------------------------

# The three full–size judge models.
JUDGES = ['claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'gpt-4o-2024-08-06']
# Hard–coded mapping from judge to provider.
JUDGE_PROVIDERS = {
    'claude-3-5-sonnet-20241022': 'anthropic',
    'gemini-1.5-pro-002': 'google',
    'gpt-4o-2024-08-06': 'openai'
}
# List of known providers.
KNOWN_PROVIDERS = ['anthropic', 'google', 'openai']

# -------------------------------
# Helper Functions
# -------------------------------

def get_score_column(df, model):
    """
    Return the score (or assessment) column for a given model in the DataFrame.
    First, if the model is one of the judges, try a preferred naming.
    Otherwise, try a column named "model_score" and, if not found,
    return the first column that contains both the model name and either "score" or "assessment".
    """
    pref = {
        'claude-3-5-sonnet-20241022': 'anthropic/',
        'gemini-1.5-pro-002': 'google/',
        'gpt-4o-2024-08-06': 'openai/'
    }
    if model in pref:
        candidate = f"{pref[model]}{model}_score"
        if candidate in df.columns:
            return candidate
    candidate = f"{model}_score"
    if candidate in df.columns:
        return candidate
    for col in df.columns:
        if (('score' in col.lower() or 'assessment' in col.lower()) and model in col):
            return col
    return None

def mean_se_ci_str(values):
    """
    Given a list of numeric values, compute the mean, standard error (SE)
    and 95% confidence interval.
    Returns a string formatted as: Mean (SE) [CI_lower, CI_upper].
    """
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 1:
        return "NaN"
    if len(arr) == 1:
        return f"{arr[0]:.3f} (NaN) [NaN, NaN]"
    m = arr.mean()
    sd = arr.std(ddof=1)
    se = sd / np.sqrt(len(arr))
    ci_lower = m - 1.96 * se
    ci_upper = m + 1.96 * se
    return f"{m:.3f} ({se:.3f}) [{ci_lower:.3f}, {ci_upper:.3f}]"

def compute_krippendorff_alpha(judge_scores):
    """
    Compute Krippendorff's alpha given a 2D array (list of lists) of judge scores.
    """
    reliability_data = np.array(judge_scores)
    return krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='interval')

def calculate_judge_correlations(all_files):
    """
    Extract scores from each judge across all files and compute Pearson correlations
    (for each pair) as well as overall Krippendorff's alpha.
    Returns a tuple: (list_of_pair_tuples, overall_alpha, counts)
    where each pair tuple is (pair_name, n, correlation).
    Also returns counts as a dict mapping judge -> total number of observations.
    """
    judge_scores = defaultdict(list)
    for model_files in all_files.values():
        for fpath in model_files.values():
            df = pd.read_csv(fpath)
            for _, row in df.iterrows():
                for j in JUDGES:
                    col = get_score_column(df, j)
                    if col and pd.notna(row[col]):
                        try:
                            judge_scores[j].append(float(row[col]))
                        except:
                            continue
    counts = {j: len(judge_scores[j]) for j in JUDGES}
    pairs = [(JUDGES[0], JUDGES[1]), (JUDGES[0], JUDGES[2]), (JUDGES[1], JUDGES[2])]
    pair_results = []
    for j1, j2 in pairs:
        s1, s2 = np.array(judge_scores[j1]), np.array(judge_scores[j2])
        if len(s1) and len(s2):
            n_obs = min(len(s1), len(s2))
            corr, _ = pearsonr(s1[:n_obs], s2[:n_obs])
        else:
            corr = np.nan
            n_obs = 0
        pair_results.append((f"{j1} vs {j2}", n_obs, corr))
    judge_arrays = []
    if judge_scores:
        max_len = max(len(scores) for scores in judge_scores.values())
        for j in JUDGES:
            scores = judge_scores[j][:]
            scores.extend([np.nan]*(max_len - len(scores)))
            judge_arrays.append(scores)
        overall_alpha = compute_krippendorff_alpha(judge_arrays)
    else:
        overall_alpha = np.nan
    return pair_results, overall_alpha, counts

def infer_provider(model, df):
    """
    Try to infer the provider for a given model by scanning the DataFrame’s column names.
    Returns the part before the slash (lowercase) if it is in KNOWN_PROVIDERS.
    """
    for col in df.columns:
        if model in col and "/" in col:
            prov = col.split("/")[0].lower()
            if prov in KNOWN_PROVIDERS:
                return prov
    return None

def get_related_judge_for_model(model, df):
    """
    For a given model (string) and its first CSV (df), determine its related judge.
    If the model is one of the judges, return the model (i.e. it is related to itself);
    otherwise, infer its provider and return the corresponding judge from JUDGE_PROVIDERS.
    """
    if model in JUDGES:
        return model
    prov = infer_provider(model, df)
    if prov and prov in JUDGE_PROVIDERS.values():
        for judge, p in JUDGE_PROVIDERS.items():
            if p == prov:
                return judge
    return None

def compute_global_judge_harshness(all_files, judges, judge_providers):
    """
    For non–judge models, compute each judge’s global harshness as the average score that judge gives
    to models whose inferred provider is not equal to the judge’s provider.
    Returns two dictionaries mapping judge -> (numeric value, formatted string).
    """
    scores_by_judge = {j: [] for j in judges}
    for model, run_map in all_files.items():
        try:
            df_first = pd.read_csv(next(iter(run_map.values())))
        except:
            continue
        prov = infer_provider(model, df_first)
        for run_id, fpath in run_map.items():
            df = pd.read_csv(fpath)
            for _, row in df.iterrows():
                for j in judges:
                    col = get_score_column(df, j)
                    if col and pd.notna(row[col]):
                        try:
                            val = float(row[col])
                        except:
                            continue
                        if prov is not None and prov != judge_providers[j]:
                            scores_by_judge[j].append(val)
    harshness_num = {}
    harshness_fmt = {}
    for j in judges:
        if scores_by_judge[j]:
            avg = np.mean(scores_by_judge[j])
            harshness_num[j] = avg
            harshness_fmt[j] = mean_se_ci_str(scores_by_judge[j])
        else:
            harshness_num[j] = np.nan
            harshness_fmt[j] = "NaN"
    return harshness_num, harshness_fmt

def parse_mean(s):
    """
    Parse the first floating–point number from a string.
    """
    try:
        return float(s.split()[0])
    except:
        return np.nan

# -------------------------------
# Main Processing Function
# -------------------------------

def process_files(directory):
    """
    Process CSV files from the given directory. Files are grouped by model.
    
    For each model, compute:
      - Raw (unadjusted) average score (across rows and runs);
      - Adjusted average score using self–preference adjustment;
      - Self–preference bias = (Raw – Adjusted)*3;
      - Per–judge averages;
      - And Related Judge Harshness.
      
    For any model whose inferred provider is one of the judge providers the adjustment is applied.
    For judge models (where the related judge is the model itself) the adjustment is computed by
      – Taking the scores from the other judges (“rec_vals”) in the same row,
      – Looking up (if possible) what those other judges gave when evaluating this model (the “giv” values),
      – Falling back to global harshness if no giv values are found.
    For non–judge (compact) models, the global harshness for the related judge is used.
    
    Additionally, a new column “Role” is added:
      - "JUDGE" if the model is in JUDGES;
      - "Related" if its provider is one of the judge providers (but it is not in JUDGES);
      - "independent" otherwise.
      
    The final DataFrame is re–ordered so that the columns appear in the order:
        Role, completions, n, Average Score, Unadjusted Average Score, 
        (each judge’s score), Self-Preference, Related Judge Harshness.
    Finally, the models are sorted by the Mean of the Average Score (descending).
    
    Returns (results DataFrame, correlations, overall_alpha, ranking_alpha).
    """
    # Group files by model.
    all_files = defaultdict(lambda: defaultdict(str))
    for fpath in glob.glob(os.path.join(directory, 'results_*.csv')):
        base = os.path.basename(fpath).replace('results_', '').replace('.csv', '')
        if '_run' in base:
            *modelparts, run = base.split('_run')
            key = '_run'.join(modelparts)
            all_files[key][run] = fpath
        else:
            all_files[base]['1'] = fpath

    # Compute global judge harshness (for non–judge models).
    global_harshness_num, global_harshness_fmt = compute_global_judge_harshness(all_files, JUDGES, JUDGE_PROVIDERS)
    # Compute overall judge correlations and Krippendorff's alpha (for raw scores).
    pair_results, overall_corr_alpha, global_counts = calculate_judge_correlations(all_files)
    
    results = {}
    
    for model, run_map in all_files.items():
        try:
            df_first = pd.read_csv(next(iter(run_map.values())))
        except Exception as e:
            continue
        related_judge = get_related_judge_for_model(model, df_first)
        is_judge = (model in JUDGES)
        
        # Determine Role.
        if is_judge:
            role = "JUDGE"
        elif related_judge is not None:
            role = "Related"
        else:
            role = "independent"
        
        raw_scores_all = []      # Accumulate raw scores (per row).
        adjusted_scores_all = [] # Accumulate adjusted scores (per row).
        self_bias_all = []       # Accumulate self–preference bias (per row).
        judge_scores_data = {j: [] for j in JUDGES}  # Per–judge averages.
        giv_all = []  # For judge models, accumulate all giv values.
        total_rows = 0
        
        for run_id, fpath in run_map.items():
            try:
                df = pd.read_csv(fpath)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue
            for idx, row in df.iterrows():
                row_judge_scores = {}
                for j in JUDGES:
                    col = get_score_column(df, j)
                    if col and pd.notna(row[col]):
                        try:
                            val = float(row[col])
                        except:
                            continue
                        row_judge_scores[j] = val
                        judge_scores_data[j].append(val)
                if row_judge_scores:
                    raw_val = np.mean(list(row_judge_scores.values()))
                else:
                    raw_val = np.nan
                
                if related_judge is not None:
                    if is_judge:
                        # For judge models, use the other judges' scores as rec_vals.
                        other_judges = [oj for oj in JUDGES if oj != model]
                        rec_vals = [row_judge_scores[oj] for oj in other_judges if oj in row_judge_scores]
                        giv_vals = []
                        for oj in other_judges:
                            # Try to look up what score judge oj gave to model in the file for judge oj for the same run.
                            if oj in all_files and run_id in all_files[oj]:
                                try:
                                    df2 = pd.read_csv(all_files[oj][run_id][0])
                                    col2 = get_score_column(df2, model)
                                    if col2 and col2 in df2.columns:
                                        vals = df2[col2].dropna().astype(float).values
                                        if len(vals) > 0:
                                            giv_vals.append(np.mean(vals))
                                except Exception as e:
                                    continue
                        if not giv_vals:
                            # Fallback: use global harshness for the evaluated judge (i.e. itself).
                            fallback = global_harshness_num.get(model, np.nan)
                            if not np.isnan(fallback):
                                giv_vals = [fallback]
                        if rec_vals and giv_vals:
                            adjusted_val = (sum(rec_vals) + np.mean(giv_vals)) / 3.0
                        elif rec_vals:
                            fallback = global_harshness_num.get(model, np.nan)
                            adjusted_val = (sum(rec_vals) + fallback) / 3.0 if not np.isnan(fallback) else np.nan
                        else:
                            adjusted_val = np.nan
                        if giv_vals:
                            giv_all.extend(giv_vals)
                    else:
                        # For non–judge (compact) models: use the two judges other than the related judge.
                        other_judges = [oj for oj in JUDGES if oj != related_judge]
                        rec_vals = [row_judge_scores[oj] for oj in other_judges if oj in row_judge_scores]
                        H = global_harshness_num.get(related_judge, np.nan)
                        if rec_vals and not np.isnan(H):
                            adjusted_val = (sum(rec_vals) + H) / 3.0
                        else:
                            adjusted_val = np.nan
                else:
                    adjusted_val = raw_val
                
                if (adjusted_val is not None) and (not np.isnan(adjusted_val)):
                    self_bias = (raw_val - adjusted_val) * 3
                else:
                    self_bias = np.nan
                
                raw_scores_all.append(raw_val)
                adjusted_scores_all.append(adjusted_val)
                self_bias_all.append(self_bias)
                total_rows += 1
        
        completions = len(run_map)
        n_points = total_rows
        outrow = {
            "Role": role,
            "completions": completions,
            "n": n_points,
            "Unadjusted Average Score": mean_se_ci_str(raw_scores_all)
        }
        for j in JUDGES:
            outrow[j] = mean_se_ci_str(judge_scores_data[j])
        # Place Average Score (adjusted) next.
        if related_judge is not None:
            outrow["Average Score"] = mean_se_ci_str(adjusted_scores_all)
            outrow["Self-Preference"] = mean_se_ci_str(self_bias_all)
            if is_judge:
                outrow["Related Judge Harshness"] = mean_se_ci_str(giv_all) if giv_all and len(giv_all) else "n/a"
            else:
                outrow["Related Judge Harshness"] = global_harshness_fmt.get(related_judge, "n/a")
        else:
            outrow["Average Score"] = outrow["Unadjusted Average Score"]
            outrow["Self-Preference"] = "n/a"
            outrow["Related Judge Harshness"] = "n/a"
        results[model] = outrow

    df_res = pd.DataFrame(results).T

    # --- Reorder Columns ---
    # Desired order: Role, completions, n, Average Score, Unadjusted Average Score, 
    # then each judge's column, then Self-Preference, then Related Judge Harshness.
    desired_order = ["Role", "completions", "n", "Average Score", "Unadjusted Average Score"] + \
                    JUDGES + ["Self-Preference", "Related Judge Harshness"]
    # Only include columns that exist.
    desired_order = [col for col in desired_order if col in df_res.columns]
    df_res = df_res[desired_order]

    # --- Sort Models by Mean of "Average Score" (descending) ---
    df_res["AvgNumeric"] = df_res["Average Score"].apply(parse_mean)
    df_res = df_res.sort_values(by="AvgNumeric", ascending=False)
    df_res = df_res.drop(columns=["AvgNumeric"])

    # --- Compute a Weighted Average Row ---
    def wavg_and_se_ci(col, df):
        pts = []
        for idx, row in df.iterrows():
            try:
                n_val = int(row["n"])
            except:
                continue
            m_val = parse_mean(row.get(col, "NaN"))
            if not np.isnan(m_val):
                pts.extend([m_val] * n_val)
        if not pts:
            return "NaN"
        arr = np.array(pts, dtype=float)
        m = arr.mean()
        sd = arr.std(ddof=1)
        se = sd / np.sqrt(len(arr))
        ci_lower = m - 1.96 * se
        ci_upper = m + 1.96 * se
        return f"{m:.3f} ({se:.3f}) [{ci_lower:.3f}, {ci_upper:.3f}]"

    w_row = {"Role": "", "completions": "", "n": df_res["n"].sum()}
    for col in df_res.columns:
        if col not in ["Role", "completions", "n"]:
            w_row[col] = wavg_and_se_ci(col, df=df_res)
    df_res.loc["Weighted Average"] = w_row

    # --- Judge Correlations and Ranking Correlations ---
    # Recompute global judge scores (for correlation calculation) using calculate_judge_correlations:
    pair_results, overall_corr_alpha, counts = calculate_judge_correlations(all_files)
    # Build a table (list of dicts) with columns: Judge Pair, n, Correlation, Ranking Correlation.
    corr_table = []
    # For ranking correlations, we use the final results table.
    for (pair_name, n_obs, corr_val) in pair_results:
        j1, j2 = pair_name.split(" vs ")
        # Extract parsed judge means from df_res (all models except the weighted average row).
        # We assume that for each model, df_res[j] is a string like "x (se) [ci_lower, ci_upper]".
        x = df_res.loc[df_res.index != "Weighted Average", j1].apply(parse_mean).values
        y = df_res.loc[df_res.index != "Weighted Average", j2].apply(parse_mean).values
        if len(x) > 0 and len(y) > 0:
            rank_corr, _ = spearmanr(x, y)
        else:
            rank_corr = np.nan
        corr_table.append({
            "Judge Pair": pair_name,
            "n": n_obs,
            "Correlation": corr_val,
            "Ranking Correlation": rank_corr
        })
    corr_df = pd.DataFrame(corr_table)
    
    # Compute overall ranking alpha:
    # Build a matrix: rows = judges, columns = models (using parsed judge mean scores from df_res)
    ranking_matrix = []
    models_for_rank = df_res.index[df_res.index != "Weighted Average"]
    for j in JUDGES:
        scores = df_res.loc[models_for_rank, j].apply(parse_mean).values
        # Compute ranks for these scores.
        if len(scores) > 0:
            ranks = rankdata(scores)
        else:
            ranks = np.array([])
        ranking_matrix.append(ranks)
    if ranking_matrix and ranking_matrix[0].size > 0:
        overall_rank_alpha = compute_krippendorff_alpha(ranking_matrix)
    else:
        overall_rank_alpha = np.nan

    # -------------------------------
    # Main Function Return
    # -------------------------------
    return df_res, corr_df, overall_corr_alpha, overall_rank_alpha

# -------------------------------
# Main Function
# -------------------------------

def main(dir_):
    # Set pandas display options.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    
    df_res, corr_df, overall_corr_alpha, overall_rank_alpha = process_files(dir_)
    
    print("\nLLM-as-a-Judge scores with pooled runs.")
    print("Each score is reported as: Mean (SE) [95% CI]\n")
    print(df_res)
    
    print("\nJudge Correlations and Ranking Correlations:")
    print("-" * 70)
    # Format the correlation table with 3 columns: Judge Pair, n, Correlation, Ranking Correlation.
    print(corr_df.to_string(index=False, float_format="%.3f"))
    print("-" * 70)
    print(f"Overall Krippendorff's Alpha (Correlation): {overall_corr_alpha:.3f}")
    print(f"Overall Krippendorff's Alpha (Ranking):   {overall_rank_alpha:.3f}")
    print("-" * 70 + "\n")

if __name__ == "__main__":
    # Adjust the directory path as needed.
    main("/content/drive/MyDrive/eval_outputs/")
