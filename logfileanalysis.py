import os
import csv
import numpy as np
import krippendorff
import sys
import re
import shutil
from datetime import datetime
from inspect_ai.log import list_eval_logs, read_eval_log
from pathlib import Path
from urllib.parse import unquote

DEFAULT_SCORE = 0

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_latest_log_file(log_dir="./logs"):
    """Find the most recent .eval log file in the logs directory"""
    logs = list_eval_logs(log_dir, recursive=False)
    if not logs:
        print(f"No logs found in {log_dir}")
        return None
    latest = sorted(logs, key=lambda x: os.path.basename(x.name))[-1]
    log_path = latest.name
    return log_path

def extract_risk_category(text):
    lines = text.strip().split('\n')
    first_line = lines[0] if lines else ''
    match = re.match(r'^\[([A-E])\]', first_line)
    return match.group(1) if match else None

def compute_alpha_safely(data, level):
    try:
        return krippendorff.alpha(reliability_data=data, level_of_measurement=level)
    except ValueError as e:
        print(f"Warning: Could not compute alpha for {level}: {e}")
        print(f"Unique values in data: {np.unique(data[~np.isnan(data)])}")
        return float('nan')

def extract_scores_and_compute_alpha(log_file=None, log_dir="./logs", output_dir="./outputs"):
    """
    Extract scores and compute alpha coefficient from evaluation logs.
    Args:
        log_file: Optional specific log file to analyze
        log_dir: Directory containing log files (default: ./logs)
        output_dir: Directory to save results (default: ./outputs)
    """
    if log_file:
        log_file_path = log_file
    else:
        log_file_path = get_latest_log_file(log_dir)
        if not log_file_path:
            print(f"No logs found in {log_dir}")
            return

    # Decode the URL-encoded filename
    log_filename = unquote(os.path.basename(log_file_path))
    decoded_log_file_path = os.path.join(os.path.dirname(log_file_path), log_filename)

    # Remove the "file://" prefix if it's still present
    if decoded_log_file_path.startswith("file://"):
        decoded_log_file_path = decoded_log_file_path[7:]

    # Copy the .eval log file to output directory with the same name
    log_backup_path = os.path.join(output_dir, log_filename)

    try:
        # Create source path without "file://" prefix
        source_path = decoded_log_file_path
        destination_path = log_backup_path

        shutil.copy2(source_path, destination_path)
        print(f"Log file backed up to {log_backup_path}")

    except Exception as e:
        print(f"Warning: Could not backup log file: {str(e)}")

    # Now read the .eval log and create CSV using the decoded path
    log = read_eval_log(decoded_log_file_path, resolve_attachments=True)
    scoring_models = [s.name for s in (log.results.scores or [])
                     if s.name not in ("avg", "error_count")]

    rows = []
    alpha_matrix_scores = []
    alpha_matrix_categories = []
    category_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

    # Process samples and compute matrices
    for sample in log.samples or []:
        answer = next((msg.content for msg in (sample.messages or [])
            if getattr(msg, "source", None) == "generate" and
            getattr(msg, "role", "") == "assistant"), "")

        scorer_data = {m: {"text": "", "category": None} for m in scoring_models}
        for ev in (sample.events or []):
            if ev.event == "model" and ev.model in scoring_models:
                try:
                    choices = ev.output.choices[0].message.content
                    text = choices[0].text if isinstance(choices, list) else choices
                    category = extract_risk_category(text.strip())
                    scorer_data[ev.model] = {
                        "text": text.strip(),
                        "category": category_mapping.get(category) if category else None
                    }
                except (AttributeError, IndexError):
                    continue

        scores = sample.scores.get("final_digit_model_graded_qa", {})
        if scores and hasattr(scores, 'value'):
            scores.value = {k: int(str(v).strip('[]')) if isinstance(v, (str, int)) and
                          str(v).strip('[]') in {'-1', '0', '1'} else DEFAULT_SCORE
                          for k, v in scores.value.items()}
        row_scores = [scores.value.get(m, DEFAULT_SCORE) if scores else DEFAULT_SCORE for m in scoring_models]
        row_categories = [scorer_data[m]["category"] for m in scoring_models]

        row = [sample.id, sample.input or "", answer]
        row.extend(scorer_data[m]["text"] for m in scoring_models)
        row.extend(next((k for k, v in category_mapping.items() if v == cat), None)
                  for cat in row_categories)
        row.extend(row_scores)

        rows.append(row)
        alpha_matrix_scores.append(row_scores)
        alpha_matrix_categories.append(row_categories)

    # Compute statistics
    arr_scores = np.array(alpha_matrix_scores, dtype=float).T
    arr_categories = np.array(alpha_matrix_categories, dtype=float).T
    alpha_scores = compute_alpha_safely(arr_scores, "interval")
    alpha_categories = compute_alpha_safely(arr_categories, "ordinal")

    # Prepare headers
    headers = (["sample_id", "question", "answer"] +
              [f"{m}_assessment" for m in scoring_models] +
              [f"{m}_category" for m in scoring_models] +
              [f"{m}_score" for m in scoring_models])

    # Ensure output directory exists
    ensure_directory(output_dir)

    # Extract timestamp from log filename
    timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\+\d{2}-\d{2})", log_filename)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # fallback

    # Save results to CSV
    csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")

    # Save CSV results
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([log.eval.model] + [""]*(len(headers)-1))
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Results saved to {csv_path}")

    # Print analysis results
    print(f"\nKrippendorff's alpha:")
    print(f"  Scores: {alpha_scores:.3f}")
    print(f"  Categories: {alpha_categories:.3f}")

    print(f"\nNumber of samples: {len(rows)}")
    for i, model_name in enumerate(scoring_models):
        scores = arr_scores[i, :]
        categories = arr_categories[i, :]
        print(f"{model_name}:")
        print(f"  Average score: {np.mean([s for s in scores if not np.isnan(s)]):.3f}")
        orig_categories = [next((k for k, v in category_mapping.items() if v == int(cat)), None)
                         for cat in categories if not np.isnan(cat)]
        category_counts = np.unique(orig_categories, return_counts=True)
        print(f"  Categories: {dict(zip(category_counts[0], category_counts[1]))}")

if __name__ == "__main__":
    args = sys.argv[1:]
    log_file = None
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
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            print(f"Error: Invalid arguments")
            sys.exit(1)

    extract_scores_and_compute_alpha(log_file=log_file, log_dir=log_dir, output_dir=output_dir)
