import pandas as pd
import json
import re

def clean_question(question):
    """Remove extra instructions from questions."""
    return re.sub(r"Your answer must be EXACTLY.*$", "", str(question)).strip()

def filter_json_by_csv_with_ids(json_path, csv_path, output_json_path):
    """Filter JSON data to include only questions present in the CSV, and add sequential question IDs starting from 1."""
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    
    # Load the CSV file
    csv_data = pd.read_csv(csv_path)

    # Clean and normalize CSV questions
    csv_questions = csv_data.iloc[1:, 1].dropna()  # Exclude the first two rows (headers)
    csv_cleaned_questions = [clean_question(q).lower() for q in csv_questions]

    # Filter the JSON data and add sequential question IDs
    filtered_json = []
    question_id = 1  # Start IDs from 1
    for entry in json_data:
        if clean_question(entry["question"]).lower() in csv_cleaned_questions:
            entry_with_id = entry.copy()
            entry_with_id["question_id"] = question_id  # Assign sequential IDs starting from 1
            filtered_json.append(entry_with_id)
            question_id += 1

    # Save the filtered JSON data
    with open(output_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(filtered_json, output_file, indent=4, ensure_ascii=False)
    
    print(f"Filtered JSON saved to {output_json_path}. Total questions: {len(filtered_json)}")

# Input paths
json_path = "/content/anai/artifacts/filtered.json"
csv_path = "/content/drive/MyDrive/eval_outputs/claude-3-5-haiku-20241022/combined_results_claude-3-5-haiku-20241022_cleaned.csv"
output_json_path = "/content/anai/artifacts/ab_reddit.json"

# Run the filtering process
filter_json_by_csv_with_ids(json_path, csv_path, output_json_path)
