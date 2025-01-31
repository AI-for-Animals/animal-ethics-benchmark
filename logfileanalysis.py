import os
import csv
import numpy as np
import sys
import re
import shutil
from datetime import datetime
from inspect_ai.log import list_eval_logs, read_eval_log
from pathlib import Path
from urllib.parse import unquote
import pandas as pd

DEFAULT_SCORE = 0

def ensure_directory(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_latest_log_file(log_dir="./logs"):
    """Find the most recent .eval log file in the logs directory."""
    logs = list_eval_logs(log_dir, recursive=False)
    if not logs:
        return None
    latest = sorted(logs, key=lambda x: os.path.basename(x.name))[-1]
    return latest.name  # filename in that directory

def extract_risk_category(text: str):
    """Return the single-letter category (A-E) from the first line, or None."""
    lines = text.strip().split('\n')
    first_line = lines[0] if lines else ''
    match = re.match(r'^\[([A-E])\]', first_line)
    return match.group(1) if match else None

def _parse_eval_log(log_file=None, log_dir="./logs"):
    """
    Parse a .eval log, returning a dictionary containing
    model_names, alpha_matrix_scores, alpha_matrix_categories, n_samples, plus a 'rows' list
    that we can write out to CSV if desired.
    """
    if log_file is None:
        # If user didn't specify, pick the latest .eval
        log_file = get_latest_log_file(log_dir)
        if not log_file:
            print(f"No .eval logs found in {log_dir}")
            return {}

    # Handle URL-decoding, remove 'file://' prefix if any
    log_file = unquote(log_file)
    if log_file.startswith("file://"):
        log_file = log_file[7:]

    # If the user gave just a filename, try combining with log_dir
    if not os.path.exists(log_file):
        candidate = os.path.join(log_dir, os.path.basename(log_file))
        if os.path.exists(candidate):
            log_file = candidate
        else:
            print(f"Log file not found: {log_file}")
            return {}

    # Read the .eval log
    log = read_eval_log(log_file, resolve_attachments=True)
    if not log or not log.samples:
        print(f"Empty or invalid log file: {log_file}")
        return {}

    # Identify which scoring models are in final_digit_model_graded_qa
    scoring_models = [
        s.name
        for s in (log.results.scores or [])
        if s.name not in ("avg", "error_count")
    ]

    category_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    model_scores = {m: [] for m in scoring_models}
    model_cats = {m: [] for m in scoring_models}

    # We'll also build CSV rows so we can write them out
    rows_for_csv = []
    n_samples = len(log.samples)

    sample_id = 0
    for sample in log.samples:
        sample_id += 1
        # Attempt to retrieve the final answer from the assistant
        final_answer = ""
        for msg in (sample.messages or []):
            if getattr(msg, "source", None) == "generate" and msg.role == "assistant":
                final_answer = msg.content
                break

        # Grab each model's text-based "category" from events
        per_model_text = {}
        for m in scoring_models:
            per_model_text[m] = ""  # will store the model's raw scoring text

        for ev in sample.events or []:
            if ev.event == "model" and ev.model in scoring_models:
                try:
                    raw = ev.output.choices[0].message.content
                    text = raw[0].text if isinstance(raw, list) else raw
                    cat_letter = extract_risk_category(text.strip())
                    per_model_text[ev.model] = text.strip()
                except:
                    pass

        # Also get numeric scores from final_digit_model_graded_qa
        final_scores = sample.scores.get("final_digit_model_graded_qa", {})
        if hasattr(final_scores, "value"):
            newvals = {}
            for k, v in final_scores.value.items():
                sval = str(v).strip("[]")
                newvals[k] = int(sval) if sval in {"-1", "0", "1"} else DEFAULT_SCORE
            final_scores = newvals
        else:
            final_scores = {}

        # Convert textual categories to numeric, store in model_cats/mode_scores
        for m in scoring_models:
            numeric_score = final_scores.get(m, DEFAULT_SCORE)
            model_scores[m].append(numeric_score)

            text_val = per_model_text[m]
            cat_letter = extract_risk_category(text_val)
            numeric_cat = category_mapping.get(cat_letter, np.nan)
            model_cats[m].append(numeric_cat)

        #
        # Build a CSV row: e.g.
        #   [sample_id, sample.input, final_answer,
        #    <model1>_assessment, <model2>_assessment, ...
        #    <model1>_category,  <model2>_category, ...
        #    <model1>_score,     <model2>_score, ...]
        #

        row = [
            sample_id,
            (sample.input or "").replace("\n", " "),
            final_answer.replace("\n", " "),
        ]

        # Add columns for each model's raw assessment text
        for m in scoring_models:
            row.append(per_model_text[m])

        # Add columns for each model's letter category
        for m in scoring_models:
            row_cats = model_cats[m]
            row_cat_letter = None
            # the cat for THIS sample is the last appended item
            if len(row_cats) == sample_id:
                c = row_cats[-1]
                if not np.isnan(c):
                    # invert the cat
                    for letter, val in category_mapping.items():
                        if val == int(c):
                            row_cat_letter = letter
                            break
            row.append(row_cat_letter if row_cat_letter else "")

        # Add columns for each model's numeric score
        for m in scoring_models:
            sc_list = model_scores[m]
            numeric_score = sc_list[-1] if len(sc_list) == sample_id else DEFAULT_SCORE
            row.append(numeric_score)

        rows_for_csv.append(row)

    # Convert them to a list-of-lists for alpha_matrix...
    alpha_matrix_scores = []
    alpha_matrix_categories = []
    # build them in the same order as scoring_models
    for m in scoring_models:
        alpha_matrix_scores.append(model_scores[m])
        alpha_matrix_categories.append(model_cats[m])

    return {
        "model_names": scoring_models,
        "alpha_scores": alpha_matrix_scores,
        "alpha_cats": alpha_matrix_categories,
        "n_samples": n_samples,
        "rows": rows_for_csv,
    }

def _parse_csv_file(csv_file):
    """
    Parse an already-combined CSV (like 'results_combined.csv') to produce the same arrays:
        model_names,
        alpha_matrix_scores,
        alpha_matrix_categories,
        n_samples
    """
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return {}

    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"No data in CSV: {csv_file}")
        return {}

    score_cols = [c for c in df.columns if c.endswith("_score")]
    if not score_cols:
        print(f"No *_score columns found in {csv_file}")
        return {}

    model_names = [c[:-6] for c in score_cols]  # remove trailing "_score"
    n_samples = len(df)
    category_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

    alpha_matrix_scores = []
    alpha_matrix_categories = []
    for model in model_names:
        scores = df[model + "_score"].fillna(DEFAULT_SCORE).astype(int).values
        cat_col = model + "_category"
        if cat_col in df.columns:
            cat_series = df[cat_col].fillna("").astype(str)
            categories_num = [category_mapping.get(x, np.nan) for x in cat_series]
        else:
            categories_num = [np.nan]*n_samples

        alpha_matrix_scores.append(scores)
        alpha_matrix_categories.append(categories_num)

    return {
        "model_names": model_names,
        "alpha_scores": alpha_matrix_scores,
        "alpha_cats": alpha_matrix_categories,
        "n_samples": n_samples,
        # rows not needed here because it's an existing CSV
        "rows": None,
    }

def _write_csv_output(rows, model_names, output_dir, log_filename=""):
    """
    Actually write 'results_{timestamp}.csv' to output_dir with your row data.
    This ensures combine_csv_files(...) will find it.
    """
    ensure_directory(output_dir)

    # Attempt to parse a timestamp from the .eval filename
    # e.g. 2025-01-31T15-18-00+00-00
    timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\+\d{2}-\d{2})", log_filename)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
    else:
        # fallback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build a header row
    headers = ["sample_id", "input", "final_answer"]
    # One assessment column per model
    for m in model_names:
        headers.append(f"{m}_assessment")
    # One category column per model
    for m in model_names:
        headers.append(f"{m}_category")
    # One score column per model
    for m in model_names:
        headers.append(f"{m}_score")

    csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Results CSV saved to: {csv_path}")

def _analyze_scores_and_categories(model_names, alpha_matrix_scores, alpha_matrix_categories, output_dir):
    """
    Given arrays (num_models x num_samples), prints:
      - # of samples
      - 'All n scorers' aggregated stats
      - each model's stats
    Then writes a short text summary to an output file.
    """
    ensure_directory(output_dir)

    num_models = len(model_names)
    if num_models == 0:
        print("No models found, nothing to analyze.")
        return

    arr_scores = np.array(alpha_matrix_scores, dtype=float)
    arr_cats = np.array(alpha_matrix_categories, dtype=float)

    n_samples = arr_scores.shape[1] if arr_scores.ndim == 2 else 0
    print(f"\nNumber of samples: {n_samples}")

    # Flatten for "All scorers" stats
    flattened_scores = arr_scores.flatten()
    flattened_scores = flattened_scores[~np.isnan(flattened_scores)]
    agg_mean = np.mean(flattened_scores) if len(flattened_scores) > 0 else float('nan')

    flattened_cats = arr_cats.flatten()
    flattened_cats = flattened_cats[~np.isnan(flattened_cats)]
    category_mapping_inv = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    flattened_letters = [category_mapping_inv.get(int(x), None) for x in flattened_cats]
    cat_counts = {}
    for letter in flattened_letters:
        if letter:
            cat_counts[letter] = cat_counts.get(letter, 0) + 1

    print(f"All {num_models} scorers:")
    print(f"  Average score: {agg_mean:.3f}")
    print(f"  Categories: {cat_counts}")

    # Per-model
    for i, m in enumerate(model_names):
        row_scores = arr_scores[i, :]
        row_cats = arr_cats[i, :]
        valid_scores = row_scores[~np.isnan(row_scores)]
        mean_score = np.mean(valid_scores) if len(valid_scores) else float('nan')

        valid_cats = row_cats[~np.isnan(row_cats)]
        letters = [category_mapping_inv.get(int(x), None) for x in valid_cats]
        dist = {}
        for lt in letters:
            if lt:
                dist[lt] = dist.get(lt, 0) + 1

        print(f"\n{m}:")
        if len(valid_scores) > 0:
            print(f"  Average score: {mean_score:.3f}")
            print(f"  Categories: {dist}")
        else:
            print("  No valid scores found.")

    # Also write a text summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"analysis_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"All {num_models} scorers:\n")
        f.write(f"  Average score: {agg_mean:.3f}\n")
        f.write(f"  Categories: {cat_counts}\n\n")
        for i, m in enumerate(model_names):
            row_scores = arr_scores[i, :]
            valid_scores = row_scores[~np.isnan(row_scores)]
            mean_score = np.mean(valid_scores) if len(valid_scores) else float('nan')

            row_cats = arr_cats[i, :]
            valid_cats = row_cats[~np.isnan(row_cats)]
            letters = [category_mapping_inv.get(int(x), None) for x in valid_cats]
            dist = {}
            for lt in letters:
                if lt:
                    dist[lt] = dist.get(lt, 0) + 1

            f.write(f"{m}:\n")
            if len(valid_scores) > 0:
                f.write(f"  Average score: {mean_score:.3f}\n")
                f.write(f"  Categories: {dist}\n\n")
            else:
                f.write("  No valid scores found.\n\n")

    print(f"\nAnalysis summary saved to: {summary_path}\n")

def main():
    args = sys.argv[1:]
    log_file = None
    csv_file = None
    log_dir = "./logs"
    output_dir = "./outputs"

    i = 0
    while i < len(args):
        if args[i] == "--log-file" and i + 1 < len(args):
            log_file = args[i + 1]
            i += 2
        elif args[i] == "--log-dir" and i + 1 < len(args):
            log_dir = args[i + 1]
            i += 2
        elif args[i] == "--csv-file" and i + 1 < len(args):
            csv_file = args[i + 1]
            i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            print(f"Error: Invalid argument or missing value near '{args[i]}'")
            sys.exit(1)

    if csv_file:
        # Parse from an already-combined CSV
        parsed = _parse_csv_file(csv_file)
        if parsed and parsed.get("n_samples", 0) > 0:
            _analyze_scores_and_categories(
                parsed["model_names"],
                parsed["alpha_scores"],
                parsed["alpha_cats"],
                output_dir
            )
    else:
        # Parse from a .eval log => also produce results_*.csv
        parsed = _parse_eval_log(log_file=log_file, log_dir=log_dir)
        if parsed and parsed.get("n_samples", 0) > 0:
            # 1) Write the CSV (so aha.py can later combine them)
            _write_csv_output(
                rows=parsed["rows"],
                model_names=parsed["model_names"],
                output_dir=output_dir,
                log_filename=os.path.basename(log_file) if log_file else "",
            )
            # 2) Run the analysis & print text summary
            _analyze_scores_and_categories(
                parsed["model_names"],
                parsed["alpha_scores"],
                parsed["alpha_cats"],
                output_dir
            )

if __name__ == "__main__":
    main()
