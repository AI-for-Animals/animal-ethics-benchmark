from typing import Any
from inspect_ai import task, Task
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import choice

@task
def anai() -> Task:
    dataset = json_dataset(
        'anai/artifacts/curated.json',
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=multiple_choice(),
        scorer=choice(),
    )
if __name__ == "__main__":
    anai()

def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record['scenario'] + ' ' + record['question'],
        choices = [
            record['option_a'],
            record['option_b'],
        ],
        target="A" if record['correct_option'] == 'option_a' else "B",
    )
