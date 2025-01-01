import json
from sentence_transformers import SentenceTransformer


with open('reddit_curated.json') as f:
    data = json.load(f)
questions = list(map(lambda par: par["question"], data))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(questions)
scores = model.similarity(embeddings, embeddings)

unique_questions = []

for i in range(len(scores)): 
    is_unique = True

    for j in range(0,i): 
        if scores[i,j] > 0.8:
            print("FOUND SEMANTICALLY EQUIVALENT QUESTIONS")
            print(f"Question 1: {questions[i]}")
            print(f"Question 2: {questions[j]}")
            is_unique = False
    if is_unique:
        unique_questions.append(data[i])
print(f"Unique: {len(unique_questions)} Original: {len(questions)}")
json.dump(unique_questions, open('filtered.json', 'w+'), indent=4)