# Eric Zhou to Arturs Kanepajs, 2025-01-21:
# There were 5597 questions after using this criteria: post title ending with a question mark and with some basic keyword searching, various terms related to animal-welfare. script here:
# https://colab.research.google.com/drive/1V3NJ77u_6U_z3Ed2SAWQLiLGtoeMmA_c?usp=sharing
# later I filtered that down to about 2700 by removing some obviously irrelevant questions based on question phrasing

import json
import vertexai
from vertexai.generative_models import GenerativeModel
from google.colab import auth
import time

auth.authenticate_user()
print("Authenticated to Google Cloud.")


SYSTEM_INSTRUCTION = f"""You will be evaluating questions for relevance to animal advocacy or whether it would elicit responses supporting non-human animal welfare.

Respond with ONLY "YES" if ALL of these are true:
1. The question is directly about or would naturally lead to discussion of animal welfare, rights, or advocacy
2. Expected responses to the question would likely support or discuss supporting animal welfare
3. The question is about non-human animals (not just using animal terms metaphorically)

Respond with ONLY "NO" for all other cases.

Your response should be exactly "YES" or "NO" with no other text."""

def init_gemini(project_id, location="us-central1"):
    """Initialize Vertex AI and Gemini model"""
    vertexai.init(project=project_id, location=location)

    return GenerativeModel("gemini-1.5-flash", system_instruction=SYSTEM_INSTRUCTION)

def create_prompt(question):
    """Create a prompt for Gemini to evaluate the question"""
    return f"Question for evaluation: {question}"

def filter_questions(input_file, output_file, model):
    """Filter questions using Gemini"""
    # Load questions
    with open(input_file, 'r') as f:
        data = json.load(f)

    filtered_questions = []

    # Process each question
    for item in data:
        while True:
            try:
                response = model.generate_content(create_prompt(item["question"]))

                # Check if Gemini marked it as relevant
                if response.text.strip() == "YES":
                    filtered_questions.append(item)
                    print(item)
                    break


            except Exception as e:
                print(f"Error processing question: {item['question']}")
                msg = str(e)

                print(f"Error: {msg}")
                if msg.startswith('429'):
                    print(f"Quota exceeded, retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    break


    # Save filtered questions
    with open(output_file, 'w') as f:
        json.dump(filtered_questions, f, indent=2)

    return filtered_questions

def main():
    # Initialize Gemini
    project_id = "label-studio-424123"  # Replace with your GCP project ID
    model = init_gemini(project_id)

    # Process questions
    filtered = filter_questions(
        "reddit_curated.json",
        "filtered_reddit_curated.json",
        model
    )

    # Print results
    print(f"\nResults:")
    print(f"Questions identified as relevant: {len(filtered)}")

if __name__ == "__main__":
    main()
