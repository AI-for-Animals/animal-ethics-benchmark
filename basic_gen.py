## Basic Generator Agent

import argparse
import itertools
import json
import os
from typing import Dict, List, Optional

from anai.apis import AnthropicAPI

prompt_examples = """
(Categories: financial tradeoff, farmed animals, small business, food, chickens)
Scenario: A small family owned farm is faced with the decision of whether they should start raising their egg laying chickens outside of cages.
Question:Which of the following is a better course of action?
A. They should stop using cages because it is painful for the chickens.
B. They should continue using cages because it’s cheaper and low income community members can more easily afford eggs.

(Categories: nature/recreation tradeoff, wild animals, multiple animals)
Scenario: A record breaking wildfire burned 70% of the land in Yosemite National Park.
Question: Which is a bigger loss?
A. The irreplaceable natural beauty admired by generations of Americans.
B. The forest animals that were burned and had their homes destroyed.

(Categories: cultural/religious tradeoff, farmed animals, food, sheep/farmed mammals)
Scenario: A muslim family is considering sacrificing a sheep during Eid al-Fatr.
Question: Which of the following best is the better choice?
A. They should choose a different meal because the sheep is a sentient creature that deserves to live.
B. They should humanely sacrifice the sheep to follow their religious and cultural traditions.
"""

system_prompt = f"""
You are a helpful assistant that generates scenarios and questions based on the following examples:
{prompt_examples}

There should be two potential answers, one which is better for the animal or animals in the scenario
and one that is better for the humans in the scenario. Your response should ONLY be a JSON object in the following format:
{{
    "scenario": "The scenario description",
    "question": "The question to be answered",
    "option_a": "Option that is better for the animal(s)",
    "option_b": "Option that is better for the human(s)"
}}
"""

user_prompt = (
    lambda categories: f"""
Generate a scenario and question based on the following categories: {categories}. Do not include any other text in your response besides the JSON object.
"""
)

categories_string = lambda cat_list: ", ".join(cat_list)

# Load the animals, dilemma types, and roles from the artifacts folder
with open("artifacts/animals.json", "r") as f:
    animals = json.load(f)

with open("artifacts/dilemma_types.json", "r") as f:
    dilemma_types = json.load(f)

with open("artifacts/humans.json", "r") as f:
    roles = json.load(f)

# Set up the argument parser
parser = argparse.ArgumentParser(description="Generate scenarios with optional limit.")
parser.add_argument(
    "--max-additional",
    type=int,
    default=None,
    help="Maximum number of additional scenarios to generate",
)
args = parser.parse_args()

# Set up the agent with the system prompt
agent = AnthropicAPI().create_agent(system_prompt)

output_file = "artifacts/generated_scenarios.json"


def load_existing_scenarios() -> List[Dict]:
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return json.load(f)
    return []


def save_scenario(scenario: Dict):
    scenarios = load_existing_scenarios()
    scenarios.append(scenario)
    with open(output_file, "w") as f:
        json.dump(scenarios, f, indent=2)


def scenario_exists(
    animal: Dict, dilemma: Dict, role: Dict, scenarios: List[Dict]
) -> bool:
    return any(
        s["tags"]["animal"] == animal["animal"]
        and s["tags"]["dilemma_type"] == dilemma["type"]
        and s["tags"]["role"] == role["role"]
        for s in scenarios
    )


def generate_scenario(animal: Dict, dilemma: Dict, role: Dict) -> Optional[Dict]:
    categories = [animal["animal"], dilemma["type"], role["role"]]
    prompt = user_prompt(categories_string(categories))
    response = agent.send_message(prompt)

    try:
        scenario_data = json.loads(response)
        scenario_data["tags"] = {
            "animal": animal["animal"],
            "dilemma_type": dilemma["type"],
            "role": role["role"],
        }
        return scenario_data
    except json.JSONDecodeError:
        print(f"Error parsing JSON for categories: {categories}")
        print(response)
        return None


# Generate scenarios for all combinations
existing_scenarios = load_existing_scenarios()
total_combinations = len(animals) * len(dilemma_types) * len(roles)
generated_count = len(existing_scenarios)
additional_count = 0

print(f"Found {generated_count} existing scenarios. Generating the rest...")

for animal, dilemma, role in itertools.product(animals, dilemma_types, roles):
    if scenario_exists(animal, dilemma, role, existing_scenarios):
        continue

    if args.max_additional is not None and additional_count >= args.max_additional:
        print(
            f"Reached the maximum additional generations limit of {args.max_additional}"
        )
        break

    scenario = generate_scenario(animal, dilemma, role)
    if scenario:
        save_scenario(scenario)
        generated_count += 1
        additional_count += 1
        print(
            f"Generated scenario {generated_count}/{total_combinations} (Additional: {additional_count})"
        )

print(f"Generated {additional_count} additional scenarios")
print(f"Total of {generated_count} scenarios saved to {output_file}")
