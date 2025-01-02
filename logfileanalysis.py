import os
import csv
import numpy as np
import krippendorff 
import sys
import re
from datetime import datetime
from inspect_ai.log import list_eval_logs, read_eval_log

DEFAULT_SCORE = 0

# Rest of the code remains the same

def get_latest_log_file(log_dir="./logs"):
   logs = list_eval_logs(log_dir, recursive=False)
   return sorted(logs, key=lambda x: os.path.basename(x.name))[-1] if logs else None

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

def extract_scores_and_compute_alpha(log_file=None, log_dir="./logs"):
   if log_file:
       log_file_path = log_file
   else:
       info = get_latest_log_file(log_dir)
       if not info:
           print(f"No logs found in {log_dir}")
           return
       log_file_path = info.name

   log = read_eval_log(log_file_path, resolve_attachments=True)
   scoring_models = [s.name for s in (log.results.scores or [])
                    if s.name not in ("avg", "error_count")]

   rows = []
   alpha_matrix_scores = []
   alpha_matrix_categories = []
   category_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

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

   bn = os.path.basename(log_file_path)
   timestamp = bn.split("_anai-open-ended_")[0]
   output_csv = f"./results_{timestamp}.csv"

   headers = (["sample_id", "question", "answer"] + 
             [f"{m}_assessment" for m in scoring_models] +
             [f"{m}_category" for m in scoring_models] +
             [f"{m}_score" for m in scoring_models])

   with open(output_csv, "w", newline="", encoding="utf-8") as f:
       writer = csv.writer(f)
       writer.writerow([log.eval.model] + [""]*(len(headers)-1))
       writer.writerow(headers)
       writer.writerows(rows)

   arr_scores = np.array(alpha_matrix_scores, dtype=float).T
   arr_categories = np.array(alpha_matrix_categories, dtype=float).T

   alpha_scores = compute_alpha_safely(arr_scores, "interval")
   alpha_categories = compute_alpha_safely(arr_categories, "ordinal")

   print(f"Results saved to {output_csv}")
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
   
   i = 0
   while i < len(args):
       if args[i] == "--log-file" and i + 1 < len(args):
           log_file = args[i + 1]
           i += 2
       elif args[i] == "--log-dir" and i + 1 < len(args):
           log_dir = args[i + 1]
           i += 2
       else:
           print(f"Error: Invalid arguments")
           sys.exit(1)
   
   extract_scores_and_compute_alpha(log_file=log_file, log_dir=log_dir)
