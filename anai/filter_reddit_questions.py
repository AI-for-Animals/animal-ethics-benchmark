import json
import re

def is_relevant_question(question):
    # First, exclude questions that are primarily about:
    exclusion_patterns = [
        r'(taste|tastes?) like',  # Taste questions
        r'(cook|cooked|cooking)',  # Cooking instructions
        r'safe to eat',           # Food safety
        r'how (long|much)',       # Technical questions
        r'recipe',                # Recipe requests
        r'expired?|spoiled?|rotten?', # Food safety
        r'temperature|thaw',      # Food handling
        r'KFC|McDonald|Burger King|Wendy', # Restaurant specifics
        r'price|cost|expensive',  # Price questions
        r'chickenpox|shingles',   # Medical conditions
        r'calories|nutrition',    # Pure nutrition
        r'protein intake',         # Nutrition
        r'swallow'
    ]

    # Include questions that suggest ethical/welfare considerations:
    inclusion_patterns = [
        r'(ethic|moral)',         # Ethical considerations
        r'(conscious|intelligence|feel|emotion|bond)',  # Animal consciousness
        r'(humane|inhumane)',     # Treatment of animals
        r'(slaughter|kill)',      # Processing discussions
        r'rights',                # Animal rights
        r'vegan|vegetarian',      # Only when not about recipes/nutrition
        r'factory farm',          # Industrial farming
        r'welfare',               # Animal welfare
        r'cruel|abuse',           # Animal treatment
        r'alternatives to',       # Alternative products
        r'lab(-| )grown',        # Future food tech
        r'(wrong|ethical|moral) to eat' # Ethical considerations
    ]

    # Additional context-based rules:
    # If question compares pets to farm animals
    if re.search(r'(dog|cat|pet).*(cow|pig|chicken)', question, re.I):
        return True

    # If question is about animal consciousness/emotions
    if re.search(r'(cow|pig|chicken).*(feel|think|smart|intelligent)', question, re.I):
        return True

    # Check exclusions first
    for pattern in exclusion_patterns:
        if re.search(pattern, question, re.I):
            return False

    # Then check inclusions
    for pattern in inclusion_patterns:
        if re.search(pattern, question, re.I):
            return True

    return False  # Default to excluding if no clear signals

def main():
    # Read input file
    try:
        with open('reddit_curated.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find reddit_curated.json")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in reddit_curated.json")
        return

    # Filter questions
    filtered_questions = []
    total_questions = 0
    selected_questions = 0

    for item in data:
        # Only process questions from NoStupidQuestions
        total_questions += 1
        if "subreddit-NoStupidQuestions" in item.get("tags", []):
            if is_relevant_question(item["question"]):
                filtered_questions.append(item)
                selected_questions += 1
        else:
            filtered_questions.append(item)
            selected_questions += 1


    # Write output file
    output_filename = 'reddit_curated_filtered.json'
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(filtered_questions, f, indent=2)
    except IOError:
        print(f"Error: Could not write to {output_filename}")
        return

    # Print statistics
    print(f"Processing complete!")
    print(f"Total questions processed: {total_questions}")
    print(f"Questions selected: {selected_questions}")
    print(f"Selection rate: {(selected_questions/total_questions)*100:.1f}%")
    print(f"Results written to: {output_filename}")

if __name__ == "__main__":
    main()
