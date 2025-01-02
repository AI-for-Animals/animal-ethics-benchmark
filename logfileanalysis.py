import os
import csv
import numpy as np
import krippendorff
import sys
from datetime import datetime

from inspect_ai.log import (
    list_eval_logs,
    read_eval_log,
    resolve_sample_attachments,  # We might also call this per-sample if needed
)

def parse_timestamp_from_filename(path: str) -> str:
    bn = os.path.basename(path)
    prefix = bn.split("_anai-open-ended_")[0]
    return prefix

def get_latest_log_file(log_dir="./logs"):
    logs = list_eval_logs(log_dir, recursive=False)
    if not logs:
        return None

    def sort_key(info):
        bn = os.path.basename(info.name)
        prefix = bn.split("_anai-open-ended_")[0]
        dt_str = prefix.split('+')[0]
        try:
            return datetime.strptime(dt_str, "%Y-%m-%dT%H-%M-%S")
        except:
            return datetime.min

    sorted_logs = sorted(logs, key=sort_key)
    return sorted_logs[-1]


def extract_scores_and_compute_alpha(log_file=None, log_dir="./logs"):
    """
    Reads the given (or latest) .eval file, with attachments resolved so that
    large blocks of content (answer text, scorer text) become visible.

    Then:
      - Extracts user question (sample.input)
      - Extracts final assistant answer from sample.messages
      - Extracts each scoring model’s textual assessment from sample.events
      - Excludes 'avg' and 'error_count' from scoring models
      - Writes to CSV -> (sample_id, question, answer, [model]_assessment..., [model]_score...)
      - Computes Krippendorff's alpha over the numeric scores.
    """
    if log_file:
        log_file_path = log_file
        print(f"Reading specified log file: {log_file_path}")
    else:
        info = get_latest_log_file(log_dir)
        if not info:
            print(f"No logs found in {log_dir}. Nothing to extract.")
            return
        log_file_path = info.name
        print(f"Reading newest log file in {log_dir}: {log_file_path}")

    # ----------------------------------------------------------------
    # 1) Read the log *with attachments resolved*
    # ----------------------------------------------------------------
    log = read_eval_log(log_file_path, resolve_attachments=True)
    if log.status != "success":
        print(f"Warning: log has status={log.status} (may be incomplete).")

    heading = log.eval.model

    # ----------------------------------------------------------------
    # 2) Identify real scoring models, excluding "avg"/"error_count"
    # ----------------------------------------------------------------
    all_scoring_models = []
    if log.results and log.results.scores:
        all_scoring_models = [s.name for s in log.results.scores]
    scoring_models = [m for m in all_scoring_models if m not in ("avg","error_count")]

    rows = []
    alpha_matrix = []

    # ----------------------------------------------------------------
    # 3) For each sample, gather question, answer, assessments, numeric scores
    # ----------------------------------------------------------------
    for sample in (log.samples or []):
        # If needed, we can also do:
        # sample = resolve_sample_attachments(sample)

        sample_id = sample.id
        question_text = sample.input or ""

        # A) Extract final assistant "answer" from sample.messages
        answer_content = ""
        for msg in (sample.messages or []):
            if getattr(msg, "source", None) == "generate" and getattr(msg, "role", "") == "assistant":
                answer_content = msg.content
                break

        # B) Extract textual assessments for each scorer from sample.events 
        scorer_contents = {model: "" for model in scoring_models}
        for ev in (sample.events or []):
            if ev.event == "model" and ev.model in scorer_contents:
                try:
                    choices = ev.output.choices[0].message.content
                    # Handle anthropic/claude style responses
                    if isinstance(choices, list):
                        content = choices[0].text
                    else:
                        content = choices
                    scorer_contents[ev.model] = content.strip()
                except (AttributeError, IndexError):
                    continue


        # C) Numeric scores from final_digit_model_graded_qa
        row_scores = []
        final_scores = sample.scores.get("final_digit_model_graded_qa", None)
        if final_scores and final_scores.value:
            row_scores = [final_scores.value.get(m, None) for m in scoring_models]
        else:
            row_scores = [None]*len(scoring_models)

        # D) Build row: ID, question, answer, each model’s text, each model’s numeric score
        row = [sample_id, question_text, answer_content]
        for m in scoring_models:
            row.append(scorer_contents[m])
        row.extend(row_scores)

        rows.append(row)
        alpha_matrix.append(row_scores)

    # ----------------------------------------------------------------
    # 4) Write CSV
    # ----------------------------------------------------------------
    timestamp_str = parse_timestamp_from_filename(log_file_path)
    output_csv = f"./results_{timestamp_str}.csv"

    headers = ["sample_id", "question", "answer"]
    headers.extend([f"{m}_assessment" for m in scoring_models])
    headers.extend([f"{m}_score" for m in scoring_models])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([heading] + [""]*(len(headers)-1))  # heading row
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

    print(f"Results saved to {output_csv}")

    # ----------------------------------------------------------------
    # 5) Compute Krippendorff's alpha
    # ----------------------------------------------------------------
    arr = np.array(alpha_matrix, dtype=float).T
    arr = np.where(np.isnan(arr), np.nan, arr)

    alpha_val = krippendorff.alpha(reliability_data=arr, level_of_measurement="interval")
    print(f"\nKrippendorff’s alpha (interval): {alpha_val:.3f}")

    # Show summary
    print(f"Number of samples: {len(rows)}")
    for i, model_name in enumerate(scoring_models):
        col = arr[i, :]
        mean_val = np.nanmean(col) if col.size > 0 else float('nan')
        print(f"{model_name}: average={mean_val:.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    log_file = None
    log_dir = "./logs"
    
    i = 0
    while i < len(args):
        if args[i] == "--log-file":
            if i + 1 < len(args):
                log_file = args[i + 1]
                i += 2
            else:
                print("Error: --log-file requires a path argument")
                sys.exit(1)
        elif args[i] == "--log-dir":
            if i + 1 < len(args):
                log_dir = args[i + 1]
                i += 2
            else:
                print("Error: --log-dir requires a path argument")
                sys.exit(1)
        else:
            print(f"Unknown argument: {args[i]}")
            sys.exit(1)
    
    extract_scores_and_compute_alpha(log_file=log_file, log_dir=log_dir)
