import pandas as pd
import glob
import os
import numpy as np
from collections import defaultdict
import re

def get_score_column(df, model):
    pref = {
        'claude-3-5-sonnet-20241022': 'anthropic/',
        'gemini-1.5-pro-002': 'google/',
        'gpt-4o-2024-08-06': 'openai/'
    }
    if model in pref:
        c = f"{pref[model]}{model}_score"
        if c in df.columns:
            return c
    b = f"{model}_score"
    if b in df.columns:
        return b
    for col in df.columns:
        if 'score' in col.lower() and model in col:
            return col
    return None

def parse_mean_sd(strval):
    """
    If the table already has an M (SD) format, we ignore it. 
    Otherwise return float(strval), np.nan if can't parse.
    """
    try:
        # e.g. "0.123 (0.456)"
        match = re.match(r'([\-\d\.]+)\s*\(([\-\d\.]+)\)', str(strval))
        if match:
            return float(match.group(1)), float(match.group(2))
        else:
            return float(strval), np.nan
    except:
        return np.nan, np.nan

def process_files(directory):
    """
    NEW APPROACH:
      1) Load all CSVs, group by model & run.
      2) For each model, gather question-level raw scores from all relevant judge columns, 
         so we can compute the unadjusted average across judges at \emph{each row}.
      3) If a model is itself a judge, also compute a question-level “Adjusted Score” 
         and “Self‐Preference” by reading the relevant columns from the other judge dataframes. 
    """
    # First, collect file paths
    # e.g. results_modelA_run1.csv => d['modelA']['1'] = [pathA1.csv]
    #      results_modelA_run2.csv => d['modelA']['2'] = [pathA2.csv], etc.
    from collections import defaultdict
    all_files = defaultdict(lambda: defaultdict(str))
    # For quickly cross-referencing: d[judge_model][run] => path
    # so we can read the judge's CSV for the same run
    for fpath in glob.glob(os.path.join(directory, 'results_*.csv')):
        base = os.path.basename(fpath).replace('results_','').replace('.csv','')
        if '_run' in base:
            # e.g. "gemini-1.5-pro-002_run2"
            *modelparts, run_id = base.split('_run')
            modelname = '_run'.join(modelparts)  # re-join if model name has '_run'
            all_files[modelname][run_id] = fpath
        else:
            # single-run with implicit run='1'
            all_files[base]['1'] = fpath

    # Judges
    judges = [
        'claude-3-5-sonnet-20241022',
        'gemini-1.5-pro-002',
        'gpt-4o-2024-08-06'
    ]

    # For final output
    results = {}

    # We will load data for each model (the "base_model") across runs:
    for base_model, run_map in all_files.items():
        # We'll accumulate question-level raw means in one big list:
        all_raw_scores = []   # unadjusted average
        # Also gather columns from each judge, so we can show mean±sd for each judge
        judge_data = defaultdict(list)  # judge_data[j] => list of question-level floats from judge j
        # If base_model is also a judge, gather question-level adjusted & selfpref
        all_adj_scores = []
        all_sp_scores = []

        # We might need to read multiple runs:
        for run_id, fpath in run_map.items():
            df_main = pd.read_csv(fpath)
            # For convenience, get the columns for all JUDGES that are present
            jcols_present = []
            for j in judges:
                col_j = get_score_column(df_main, j)
                if col_j:
                    jcols_present.append((j, col_j))

            # For question-level raw means:
            # raw_i = average of all jcols for that row
            # We'll store them in all_raw_scores
            # Also store each judge's data in judge_data[j]
            for i, row in df_main.iterrows():
                row_scores = []
                for (judge_name, colname) in jcols_present:
                    val = row[colname]
                    if pd.notna(val):
                        # store it
                        judge_data[judge_name].append(val)
                        row_scores.append(val)
                if row_scores:
                    all_raw_scores.append(np.mean(row_scores))

            # If base_model is also a judge, we compute question-level “adjusted” and “selfpref”
            # by referencing the other judges' files. We must read the relevant CSVs 
            # to see how other judges rate base_model, etc.
            if base_model in judges:
                # We'll do row by row, matching index i
                # rec_i = sum of other judges' rating of the base_model's answers
                # giv_i = how base_model rates itself in the other judge's CSV => Actually that's the "giving"? 
                #   Our original code does:
                #     rec: all other judges' rating of df_main (which is base_model's CSV)
                #     giv: how the other judges are rated by this base_model's CSV? Not quite. 
                # Actually in the old code:
                #   "rec" = means the base_model is the one being rated, so the "other" judge's columns in df_main 
                #           are how the other judge rated the base_model. 
                #   "giv" = how the base_model judged that other model in that other model's run file. 
                # For question-level, we must open each other judge's CSV for run_id 
                # and see how that DF's base_model column is. 
                # We'll do it carefully:
                other_judges = [j for j in judges if j != base_model]
                # read them now
                other_dfs = {}
                for oj in other_judges:
                    if oj in all_files and run_id in all_files[oj]:
                        other_dfs[oj] = pd.read_csv(all_files[oj][run_id])
                    else:
                        other_dfs[oj] = None

                for i, row in df_main.iterrows():
                    # rec: how the "other" judges in df_main rated base_model => i.e. jcol for that judge in df_main 
                    rec_vals = []
                    # giv: how those other_judges were rated by base_model 
                    # => we look in df for the other_judges? Actually we want base_model's column in the other judge's DF. 
                    giv_vals = []
                    for oj in other_judges:
                        oj_col = get_score_column(df_main, oj)
                        if oj_col and not pd.isna(row[oj_col]):
                            rec_vals.append(row[oj_col])
                    for oj in other_judges:
                        odf = other_dfs[oj]
                        if odf is not None:
                            # find the row i in odf => hopefully same indexing
                            # check if there's a col for base_model
                            bc = get_score_column(odf, base_model)
                            if bc is not None and i < len(odf):
                                val_g = odf.loc[i, bc]
                                if pd.notna(val_g):
                                    giv_vals.append(val_g)

                    if rec_vals and giv_vals:
                        # h = mean(giv_vals)
                        h = np.mean(giv_vals)
                        # adjusted = (sum(rec_vals)+ h) / (len(other_judges)+ 1)
                        adj_i = (sum(rec_vals) + h) / (len(other_judges) + 1)
                    else:
                        adj_i = np.nan

                    # raw_i for that question row:
                    # We'll recompute it from row_scores for that row, or we can do it again:
                    # (or store it in a dictionary with i as key)
                    # simpler to do it again quickly:
                    row_judge_vals = []
                    for (judge_name, col_j) in jcols_present:
                        val = row[col_j]
                        if pd.notna(val):
                            row_judge_vals.append(val)
                    if row_judge_vals:
                        raw_i = np.mean(row_judge_vals)
                    else:
                        raw_i = np.nan

                    if not np.isnan(adj_i) and not np.isnan(raw_i):
                        # sp = (raw_i - adj_i)*n_j (where n_j=3 if total judges=3)
                        n_j = len(judges)
                        sp_i = (raw_i - adj_i)*n_j
                    else:
                        sp_i = np.nan

                    all_adj_scores.append(adj_i)
                    all_sp_scores.append(sp_i)

        # Now we have:
        # all_raw_scores   => question-level unadjusted average (across all judges present)
        # judge_data[j]    => question-level data from judge j's column
        # all_adj_scores   => question-level data if base_model is in judges
        # all_sp_scores    => question-level data if base_model is in judges

        # Summaries
        un_label = "Unadjusted Average Score"

        def mean_sd_str(values):
            arr = np.array(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr)<2:
                if len(arr)==1:
                    return f"{arr[0]:.3f} (NaN)"
                else:
                    return "NaN"
            return f"{arr.mean():.3f} ({arr.std(ddof=1):.3f})"

        # build result row
        outrow = {}
        outrow["completions"] = len(run_map)  # number of runs
        # "n": total # of question-level data points => sum of #rows for that model
        # we can approximate that as len(all_raw_scores) for "unadjusted" but let's sum CSV lengths:
        n_points = 0
        for run_id, fpath in run_map.items():
            dtemp = pd.read_csv(fpath)
            n_points+= len(dtemp)
        outrow["n"] = n_points
        outrow[un_label] = mean_sd_str(all_raw_scores)

        # fill each judge's column
        for jg in judges:
            outrow[jg] = mean_sd_str(judge_data[jg])

        # If base_model is a judge, fill in "Average Score" and "Self-Preference"
        if base_model in judges:
            avg_label = "Average Score"
            sp_label = "Self-Preference"
            outrow[avg_label] = mean_sd_str(all_adj_scores)
            outrow[sp_label]  = mean_sd_str(all_sp_scores)
        else:
            outrow["Average Score"] = outrow[un_label]
            outrow["Self-Preference"] = "NaN"

        results[base_model] = outrow

    # put in dataframe
    df_res = pd.DataFrame(results).T

    # Weighted Average row: do it across all data points from all models
    # We'll do exactly the same approach: gather question-level data for each column
    # but let's keep it simpler: we'll do a naive weighting approach for columns that hold "M (SD)" strings
    # or just set Weighted Average to "NaN"? 
    # For completeness, let's do a naive approach that replicates the data by n points. 
    # (Same trick as before but we have to parse each string.)
    def wavg_and_sd(col, only_j=False):
        # We'll replicate by n to get ~ distribution
        big_list = []
        for idx, row in df_res.iterrows():
            if only_j and idx not in judges:
                continue
            mval, sdval = parse_mean_sd(row.get(col, "NaN"))
            # We can't recover the exact distribution from M(SD) alone, but let's approximate 
            # by a normal distribution approach. We'll do a simpler approach: we replicate "mval" n times
            # -> Weighted mean is correct but stdev is always 0 with this naive approach. 
            # That's the fundamental problem with M(SD) alone: we can't reconstruct variance 
            # from all points. So let's just do a weighted mean, and "NaN" for SD. 
            # If you truly want a "pooled stdev," you'd have to store the raw data again.
            # For a moment, let's do Weighted Mean but store "NaN" for SD, to avoid confusion. 
            if pd.isna(row.get("n", np.nan)) or np.isnan(mval):
                continue
            wcount = row["n"]
            # replicate
            big_list.extend([mval]*int(wcount))

        if not big_list:
            return "NaN"
        arr = np.array(big_list)
        return f"{arr.mean():.3f} (NaN)"

    w_row = {}
    for c in df_res.columns:
        if c=="completions":
            w_row[c] = ""
        elif c=="n":
            w_row[c] = df_res["n"].sum()
        elif c=="Self-Preference":
            w_row[c] = wavg_and_sd(c, only_j=True)
        else:
            w_row[c] = wavg_and_sd(c, only_j=False)

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
    # Example usage:
    main("/content/drive/MyDrive/eval_outputs/")
