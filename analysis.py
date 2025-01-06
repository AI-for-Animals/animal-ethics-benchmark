import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import pearsonr, norm

##############################
# CORE HELPER FUNCTIONS
##############################

def get_model_name(path):
    if 'claude' in path.lower():
        return 'Claude'
    elif 'gpt' in path.lower():
        return 'GPT'
    else:
        return 'Gemini'

def standard_error_of_mean(values):
    """
    Computes sample-based standard error of the mean (SEM):
      SEM = sqrt( var / n ),
    where var is the unbiased variance of `values`.
    """
    n = len(values)
    mean_val = np.mean(values)
    ssq = np.sum((values - mean_val)**2)
    var_ = ssq / (n - 1) if n > 1 else 0.0
    sem_ = np.sqrt(var_ / n) if n > 1 else 0.0
    return mean_val, sem_

def paired_sem(A, B):
    """
    SEM of paired difference, D_i = A_i - B_i:
      SEM(D_bar) = sqrt( sum((D_i - D_bar)^2 ) / [n*(n-1)] ).
    """
    if len(A) != len(B):
        raise ValueError("A and B must have the same length.")
    D = A - B
    n = len(D)
    d_bar = np.mean(D)
    ssq = np.sum((D - d_bar)**2)
    var_ = ssq / (n - 1) if n > 1 else 0.0
    sem_ = np.sqrt(var_ / n) if n > 1 else 0.0
    return d_bar, sem_

def compute_pairwise_comparisons_raw(models, raw_arrays):
    """
    For each pair of models (i < j), compute:
      - Means
      - Paired difference A-B (question-level)
      - SEM of the difference
      - 95% CI
      - correlation
      - p-value
    Returns a DataFrame.

    This is for *raw question-level data*.
    """
    n = len(models)
    results = []
    for i in range(n):
        mean_i = np.mean(raw_arrays[i])
        for j in range(n):
            if i >= j:
                continue
            mean_j = np.mean(raw_arrays[j])
            # Paired difference
            d_bar, sem_diff = paired_sem(raw_arrays[i], raw_arrays[j])
            ci_low = d_bar - 1.96*sem_diff
            ci_high = d_bar + 1.96*sem_diff
            # correlation
            corr = np.corrcoef(raw_arrays[i], raw_arrays[j])[0,1]
            # p-value
            z_val = d_bar / sem_diff if sem_diff>0 else 0.0
            p_val = 2*(1 - norm.cdf(abs(z_val)))

            results.append({
                "Model 1": models[i],
                "Model 2": models[j],
                "Raw Mean M1": mean_i,
                "Raw Mean M2": mean_j,
                "Diff (M1 - M2)": d_bar,
                "StdErr(Diff)": sem_diff,
                "95% CI (Diff)": (ci_low, ci_high),
                "p-value": p_val,
                "Correlation": corr,
            })

    df = pd.DataFrame(results)
    # Round numeric columns
    for col in [
        "Raw Mean M1","Raw Mean M2","Diff (M1 - M2)",
        "StdErr(Diff)","p-value","Correlation"
    ]:
        df[col] = df[col].round(4)

    df["95% CI (Diff)"] = df["95% CI (Diff)"].apply(
        lambda x: (round(x[0],4), round(x[1],4))
    )

    # Reorder columns to put "Correlation" after "p-value"
    desired_cols = [
        "Model 1", "Model 2",
        "Raw Mean M1", "Raw Mean M2",
        "Diff (M1 - M2)", "StdErr(Diff)", "95% CI (Diff)",
        "p-value", "Correlation"
    ]
    return df[desired_cols]


# NEW: Pairwise comparisons for *bootstrapped adjusted scores*
def compute_pairwise_comparisons_adjusted(models, B_adj_scores):
    """
    For each pair of models (i < j), compute the difference in adjusted scores
    over B bootstrap replicates:

      D_b = B_adj_scores[i,b] - B_adj_scores[j,b],  (b=1..B)

    Then compute:
      - Mean(D_b)
      - StdErr(D_b)
      - 95% CI
      - p-value
      - correlation of the two series B_adj_scores[i,:] vs. B_adj_scores[j,:]

    Returns a DataFrame.
    """
    n = len(models)
    B = B_adj_scores.shape[1]
    results = []
    z_95 = 1.96

    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            A = B_adj_scores[i, :]  # replicate draws for model i
            B_ = B_adj_scores[j, :] # replicate draws for model j
            D = A - B_
            d_bar = np.mean(D)
            d_std = np.std(D, ddof=1)  # unbiased stdev
            sem_diff = d_std  # Because each replicate is "one sample" for that difference

            ci_low = d_bar - z_95 * sem_diff
            ci_high= d_bar + z_95 * sem_diff

            # correlation
            corr_ij = np.corrcoef(A, B_)[0,1] if len(A)>1 else 0.0

            # p-value (two-sided)
            # (If sem_diff is 0, e.g. if replicate is degenerate, we handle that safely)
            z_val = d_bar / sem_diff if sem_diff>1e-12 else 0.0
            p_val = 2*(1 - norm.cdf(abs(z_val)))

            results.append({
                "Model 1": models[i],
                "Model 2": models[j],
                "Mean(AdjM1)": A.mean(),
                "Mean(AdjM2)": B_.mean(),
                "Diff (M1 - M2)": d_bar,
                "StdErr(Diff)": sem_diff,
                "95% CI (Diff)": (ci_low, ci_high),
                "p-value": p_val,
                "Correlation": corr_ij,
            })

    df = pd.DataFrame(results)
    # Round
    for col in [
        "Mean(AdjM1)", "Mean(AdjM2)", "Diff (M1 - M2)",
        "StdErr(Diff)", "p-value", "Correlation"
    ]:
        df[col] = df[col].round(4)
    df["95% CI (Diff)"] = df["95% CI (Diff)"].apply(
        lambda x: (round(x[0],4), round(x[1],4))
    )

    # Reorder columns so that "Correlation" is after "p-value"
    desired_cols = [
        "Model 1", "Model 2",
        "Mean(AdjM1)", "Mean(AdjM2)",
        "Diff (M1 - M2)", "StdErr(Diff)", "95% CI (Diff)",
        "p-value", "Correlation"
    ]
    return df[desired_cols]


##############################
# MAIN ANALYSIS FUNCTION
##############################

def analyze_evaluation_results(file_paths, B=200):
    """
    Analyzes the CSVs for multiple models.
    1) Builds the NxN results_matrix from the data (model i, judge j).
    2) Runs a single least-squares solver to get 'adjusted_scores' (once).
    3) Performs a bootstrap to estimate standard errors for adjusted scores.
    4) Also does pairwise comparisons for both *raw question-level data* and *adjusted bootstrap*.
    5) Plots histograms with x-axis ticks = [-1, -0.67, -0.33, 0, 0.33, 0.67, 1].

    Args:
      file_paths (List[str]): List of CSV file paths (one per model).
      B (int): Number of bootstrap replicates (default=200).
    """

    # 1) Load Data & Build NxN results_matrix
    dfs = []
    models = []

    for path in file_paths:
        mname = get_model_name(path)
        models.append(mname)
        df = pd.read_csv(path, skiprows=2)
        dfs.append(df)

    n_models = len(models)
    results_matrix = np.zeros((n_models, n_models))

    # Each df[i] => shape(#questions, ...). The last n_models columns => judge columns.
    for i in range(n_models):
        df_i_scores = dfs[i].iloc[:, -n_models:].values  # (#questions, n_models)
        mean_for_judges = df_i_scores.mean(axis=0)
        for j in range(n_models):
            results_matrix[i,j] = mean_for_judges[j]

    raw_scores = results_matrix.mean(axis=1)
    col_avgs = results_matrix.mean(axis=0)

    # 2) Solve once on the "original" results_matrix
    def objective_function(x):
        """
        x has length = 3*n_models
          x[:n_models] => judge bias
          x[n_models:2*n_models] => true_score
          x[2*n_models:] => self_bias
        We predict results_matrix[i,j] = judge_bias[j] + true_score[i] + (self_bias[i] if i==j).
        """
        n = n_models
        judge_bias = x[:n]
        true_score = x[n:2*n]
        self_bias  = x[2*n:]
        residuals = []
        for i_m in range(n):
            for j_m in range(n):
                pred = judge_bias[j_m] + true_score[i_m]
                if i_m == j_m:
                    pred += self_bias[i_m]
                residuals.append(pred - results_matrix[i_m,j_m])
        return np.array(residuals)

    x0 = np.zeros(3*n_models)
    res = least_squares(objective_function, x0)
    # unpack
    judge_scores = res.x[:n_models]
    true_scores  = res.x[n_models:2*n_models]
    self_biases  = res.x[2*n_models:]

    # shift so that mean judge bias = 0
    mean_judge = np.mean(judge_scores)
    normalized_judges = judge_scores - mean_judge
    adjusted_scores = true_scores + mean_judge

    # 3) BOOTSTRAP for ADJUSTED SCORES' STANDARD ERRORS
    # --------------------------------------------------
    # We'll do B replicates. In each replicate:
    #   - We build results_matrix_boot by sampling each df[i] with replacement
    #   - Solve the same system
    #   - Extract "adjusted_scores_boot"
    # We'll collect them in a big array of shape (n_models, B).

    def build_results_matrix_boot(dfs_original):
        """
        For each model i, sample the rows of dfs_original[i] with replacement,
        then average across each judge column => results_matrix_boot[i,j].
        """
        n = n_models
        mat_boot = np.zeros((n, n))
        for i_m in range(n):
            df_orig = dfs_original[i_m]
            nrows = len(df_orig)
            # sample with replacement
            indices = np.random.randint(0, nrows, size=nrows)
            df_i_boot = df_orig.iloc[indices, :]

            # average across the last n_models columns
            df_i_boot_vals = df_i_boot.iloc[:, -n:].values
            mean_judges_boot = df_i_boot_vals.mean(axis=0)
            for j_m in range(n):
                mat_boot[i_m, j_m] = mean_judges_boot[j_m]
        return mat_boot

    def solve_least_squares_on_matrix(mat):
        """
        Solve the same model on a NxN matrix 'mat' (like results_matrix).
        Return adjusted_scores (with mean judge bias = 0).
        """
        def obj_fun(x):
            n = n_models
            jb = x[:n]
            ts = x[n:2*n]
            sb = x[2*n:]
            resid = []
            for i_m in range(n):
                for j_m in range(n):
                    pred_ij = jb[j_m] + ts[i_m]
                    if i_m == j_m:
                        pred_ij += sb[i_m]
                    resid.append(pred_ij - mat[i_m,j_m])
            return np.array(resid)

        x0_ = np.zeros(3*n_models)
        res_ = least_squares(obj_fun, x0_)
        jb_  = res_.x[:n_models]
        ts_  = res_.x[n_models:2*n_models]
        sb_  = res_.x[2*n_models:]

        mean_jb_ = np.mean(jb_)
        adj_     = ts_ + mean_jb_
        return adj_

    B_adj_scores = np.zeros((n_models, B))  # shape: (models, replicates)

    for b_idx in range(B):
        mat_boot = build_results_matrix_boot(dfs)
        adj_boot = solve_least_squares_on_matrix(mat_boot)
        B_adj_scores[:, b_idx] = adj_boot

    # compute mean & std across replicates
    boot_means = np.mean(B_adj_scores, axis=1)
    boot_stds  = np.std(B_adj_scores, axis=1, ddof=1)  # unbiased

    # 95% CI => mean +/- 1.96 * std
    z_95 = 1.96
    boot_cis = [(m - z_95*s, m + z_95*s) for m,s in zip(boot_means, boot_stds)]

    # Build a small table for "Adjusted Scores (with bootstrap CIs)"
    df_adj_boot = pd.DataFrame({
        "Model": models,
        "Adjusted Score (Mean)": boot_means,
        "StdErr (Bootstrap)": boot_stds,
        "95% CI": boot_cis
    })
    # Round
    df_adj_boot["Adjusted Score (Mean)"] = df_adj_boot["Adjusted Score (Mean)"].round(4)
    df_adj_boot["StdErr (Bootstrap)"] = df_adj_boot["StdErr (Bootstrap)"].round(4)
    df_adj_boot["95% CI"] = df_adj_boot["95% CI"].apply(
        lambda x: (round(x[0],4), round(x[1],4))
    )

    ###################################
    # 4) PLOTS
    ###################################
    # FIGURE 1: RESULTS MATRIX
    plt.figure(figsize=(8, 6))
    plt.imshow(results_matrix, cmap='YlOrRd')
    plt.colorbar()
    for i in range(n_models):
        for j in range(n_models):
            plt.text(j, i, f'{results_matrix[i, j]:.2f}', ha='center', va='center')
    plt.xticks(range(n_models), [f"Judge: {m}" for m in models], rotation=45)
    plt.yticks(range(n_models), [f"Model: {m}" for m in models])
    plt.title("Results Matrix (Rows=Scored Models, Columns=Judges)")
    plt.tight_layout()
    plt.show()

    # FIGURE 2: HISTOGRAMS OF RAW QUESTION-LEVEL SCORES
    question_level_arrays = []
    # NEW: set x-axis ticks exactly at [-1, -0.67, -0.33, 0, 0.33, 0.67, 1]
    bin_edges = np.linspace(-1, 1, 50)  # fine bin resolution
    custom_ticks = [-1, -0.67, -0.33, 0, 0.33, 0.67, 1]

    for i, model in enumerate(models):
        df_i_scores = dfs[i].iloc[:, -n_models:].values
        row_scores = df_i_scores.mean(axis=1)
        question_level_arrays.append(row_scores)

        plt.figure(figsize=(7,4))
        plt.hist(row_scores, bins=bin_edges, alpha=0.7, edgecolor='black')
        plt.xlabel("Question-Level Scores")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Raw Scores for {model}")
        # NEW: set custom x ticks
        plt.xticks(custom_ticks, [str(t) for t in custom_ticks])
        plt.tight_layout()
        plt.show()

    # FIGURE 3: Judge & Self Bias (single solver)
    plt.figure(figsize=(8,5))
    xvals = np.arange(n_models)
    width = 0.35
    plt.bar(xvals - width/2, normalized_judges, width, label='Judge Bias (mean=0)')
    plt.bar(xvals + width/2, self_biases, width, label='Self Bias')
    plt.xticks(xvals, models, rotation=45)
    plt.legend()
    plt.title("Judge and Self Biases (Single Solver)")
    plt.tight_layout()
    plt.show()

    # FIGURE 4: RAW vs. BOOTSTRAPPED ADJUSTED (with 95% CI)
    plt.figure(figsize=(8,5))
    raw_means_plot = []
    raw_ci_low = []
    raw_ci_high= []
    for i in range(n_models):
        arr = question_level_arrays[i]
        m_i, sem_i = standard_error_of_mean(arr)
        ci_low_i = m_i - 1.96*sem_i
        ci_high_i= m_i + 1.96*sem_i
        raw_means_plot.append(m_i)
        raw_ci_low.append(ci_low_i)
        raw_ci_high.append(ci_high_i)

    xvals = np.arange(n_models)
    plt.bar(xvals - width/2, raw_means_plot, width,
            color='red', alpha=0.6, label='Raw Scores')
    plt.bar(xvals + width/2, boot_means,       width,
            color='blue', alpha=0.6, label='Adj Scores (boot)')

    for i in range(n_models):
        # raw error bars
        plt.errorbar(
            xvals[i] - width/2, raw_means_plot[i],
            yerr=[[raw_means_plot[i]-raw_ci_low[i]],
                  [raw_ci_high[i]-raw_means_plot[i]]],
            fmt='o', color='k', capsize=3
        )
        # adjusted (bootstrapped) error bars
        ci_low_i, ci_high_i = boot_cis[i]
        plt.errorbar(
            xvals[i] + width/2, boot_means[i],
            yerr=[[boot_means[i]-ci_low_i],
                  [ci_high_i-boot_means[i]]],
            fmt='o', color='k', capsize=3
        )

    plt.xticks(xvals, models, rotation=45)
    plt.title("Raw vs. Bootstrapped Adjusted Scores (95% CI)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5) PAIRWISE COMPARISONS
    #    a) RAW question-level differences
    df_pairwise_raw = compute_pairwise_comparisons_raw(models, question_level_arrays)

    #    b) ADJUSTED (BOOTSTRAPPED) differences
    df_pairwise_adj = compute_pairwise_comparisons_adjusted(models, B_adj_scores)

    # Summaries
    print("\n=== Single-Solver Adjusted Scores (No SE) ===")
    for m, val in zip(models, adjusted_scores):
        print(f"  {m}: {val:.4f}")

    print("\n=== Bootstrapped Adjusted Scores ===")
    print(df_adj_boot)

    print("\n=== Pairwise Comparisons (Paired Differences) - RAW ===")
    print(df_pairwise_raw)

    print("\n=== Pairwise Comparisons (Paired Differences) - ADJUSTED ===")
    print(df_pairwise_adj)

    return (results_matrix,
            raw_scores,
            col_avgs,
            adjusted_scores,
            normalized_judges,
            self_biases,
            df_adj_boot,           # summary of bootstrapped adjusted scores
            df_pairwise_raw,       # raw question-level pairwise
            df_pairwise_adj)       # adjusted-scores pairwise


########################################
# Example usage (comment out if needed)
########################################
if __name__ == "__main__":
    file_paths = [
        "/content/drive/MyDrive/eval_outputs/claude-3-5-haiku-20241022/combined_results_claude-3-5-haiku-20241022_cleaned.csv",
        "/content/drive/MyDrive/eval_outputs/gpt-4o-mini-2024-07-18/combined_results_gpt-4o-mini-2024-07-18_cleaned.csv",
        "/content/drive/MyDrive/eval_outputs/gemini-2.0-flash-exp/combined_results_gemini-2.0-flash-exp_cleaned.csv"
    ]
    outputs = analyze_evaluation_results(file_paths, B=1000)
