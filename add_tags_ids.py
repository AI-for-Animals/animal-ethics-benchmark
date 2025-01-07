import pandas as pd
import json
import re
import os

def clean_question(question):
    """Remove extra instructions from questions."""
    return re.sub(r"Your answer must be EXACTLY.*$", "", str(question)).strip()

def map_tags_and_ids_to_csv(json_path, csv_paths):
    """Map tags and question IDs from JSON to multiple CSV files and provide enhanced reporting."""
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    
    # Create dictionaries for mapping
    question_to_tags = {entry["question"].strip().lower(): entry["tags"] for entry in json_data}
    question_to_id = {entry["question"].strip().lower(): entry["question_id"] for entry in json_data}

    for csv_path in csv_paths:
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue

        # Load the CSV file
        csv_data = pd.read_csv(csv_path)

        # Prepare columns for tags and question IDs
        tags_column = []
        question_id_column = []
        missing_questions = []
        all_tags = {}

        for i, question in enumerate(csv_data.iloc[:, 1].dropna()):  # Assuming questions are in column B
            if i < 2:  # Skip the first two rows
                tags_column.append("")
                question_id_column.append("")
                continue

            cleaned_question = clean_question(question).lower()
            if cleaned_question in question_to_tags:
                # Append tags and question ID
                tags = question_to_tags[cleaned_question]
                tags_column.append(", ".join(tags))
                question_id_column.append(question_to_id[cleaned_question])
                # Count occurrences of each tag
                for tag in tags:
                    all_tags[tag] = all_tags.get(tag, 0) + 1
            else:
                tags_column.append("")
                question_id_column.append("")
                missing_questions.append(question)

        # Add tags and question ID columns
        csv_data['tags'] = tags_column
        csv_data['question_id'] = question_id_column

        # Save the updated CSV
        output_path = csv_path.replace(".csv", "_with_tags_and_ids.csv")
        csv_data.to_csv(output_path, index=False)
        print(f"Processed {csv_path}. Output saved to {output_path}")

        # Report
        total_questions = len(csv_data) - 2  # Exclude the first two rows
        questions_with_tags = total_questions - len(missing_questions)
        print(f"\n=== Report for {csv_path} ===")
        print(f"Total questions: {total_questions}")
        print(f"Questions with tags and IDs: {questions_with_tags}")
        print(f"Questions without tags or IDs: {len(missing_questions)}")
        print(f"Unique tags added: {len(all_tags)}")
        if all_tags:
            print("Questions per tag:")
            for tag, count in sorted(all_tags.items()):
                print(f"- {tag}: {count}")
        if missing_questions:
            print("\nQuestions without tags or IDs (up to 10 shown):")
            for mq in missing_questions[:10]:
                print(f"- {mq}")
            if len(missing_questions) > 10:
                print("... (truncated)")

# Paths to JSON and CSV files
json_path = "/content/anai/artifacts/ab_reddit.json"
csv_paths = [
    "/content/drive/MyDrive/eval_outputs/claude-3-5-haiku-20241022/combined_results_claude-3-5-haiku-20241022_cleaned.csv",
    "/content/drive/MyDrive/eval_outputs/gpt-4o-mini-2024-07-18/combined_results_gpt-4o-mini-2024-07-18_cleaned.csv",
    "/content/drive/MyDrive/eval_outputs/gemini-2.0-flash-exp/combined_results_gemini-2.0-flash-exp_cleaned.csv",
]

# Run the mapping function
map_tags_and_ids_to_csv(json_path, csv_paths)
