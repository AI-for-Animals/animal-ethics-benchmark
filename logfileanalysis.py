import json
import pandas as pd

with open('/content/logs/2024-11-01T13-56-39+00-00_anai_YYqGDwnonJwmYjxURczeeL.json', 'r') as f: # PUT THE LOG FILE PATH HERE
    document_content = f.read()

def analyze_logs(data):
    print("Analyzing log data...")

    all_results = []

    # Extract results from reductions section which contains the scoring data
    if 'reductions' in data:
        for reduction in data['reductions']:
            if reduction['scorer'] == 'choice':
                for sample in reduction['samples']:
                    result = {
                        'sample_id': sample.get('sample_id'),
                        'answer': sample.get('answer'),
                        'value': sample.get('value'),
                        'explanation': sample.get('explanation')
                    }
                    all_results.append(result)

    if not all_results:
        print("No results found in log data")
        return None

    print(f"\nProcessed {len(all_results)} results")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    print("\n=== Evaluation Summary ===")
    print(f"Total responses: {len(df)}")
    if not df.empty:
        print("\nAnswers distribution:")
        print(df['answer'].value_counts())

        correct_responses = df[df['value'] == 'C'].shape[0]
        incorrect_responses = df[df['value'] == 'I'].shape[0]
        print(f"\nCorrect responses: {correct_responses}")
        print(f"Incorrect responses: {incorrect_responses}")
        print(f"Accuracy: {(correct_responses / len(df)) * 100:.2f}%")

        # Additional metrics from the main results section
        if 'results' in data and 'scores' in data['results']:
            accuracy = data['results']['scores'][0].get('metrics', {}).get(
                'accuracy', {}).get('value')
            if accuracy:
                print(f"Overall accuracy from results: {accuracy * 100:.2f}%")

    # Calculate and print the number of answers different from A or B
    different_answers = df[~df['answer'].isin(['A', 'B'])].shape[0]
    print(f"\nAnswers different from A or B: {different_answers}")

    # Create a table with the specified format
    table_data = []
    for i in range(len(data['samples'])):
        sample = data['samples'][i]
        question_summary = sample['input'].split('\n')[0]  # Extract first line as summary
        model_answer = sample['scores']['choice']['answer']
        correct_answer = sample['target']
        correct = '✓' if model_answer == correct_answer else '✗'
        table_data.append([i + 1, question_summary, model_answer, correct_answer, correct])

    # Create DataFrame for the table
    table_df = pd.DataFrame(table_data, columns=['ID', 'Question Summary', 'Model Answer', 'Correct Answer', 'Correct?'])
    print("\n=== Question-wise Results ===")
    print(table_df.to_markdown(index=False, numalign="left", stralign="left"))

    return df

# Load and parse the JSON data
try:
    # Assuming `document_content` holds the string form of the JSON
    data = json.loads(document_content)
    results_df = analyze_logs(data)
except Exception as e:
    print(f"Error during analysis: {e}")
