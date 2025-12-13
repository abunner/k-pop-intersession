import lmstudio as lms
import os
import json
import math

llm_model = lms.llm("google/gemma-3n-e4b")
embedding_model = lms.embedding_model("text-embedding-qwen3-embedding-0.6b")

def l2_distance(a, b):
    sum = 0
    for i in range(0, len(a)):
        difference = a[i] - b[i]
        sum += difference ** 2
    return math.sqrt(sum)

with open("script_db.json", "r") as f:
    script_db = json.load(f)

with open("quiz.txt", "r") as f:
    questions = f.readlines()

input_token_count = 0
output_token_count = 0

for question in questions:
    question = question.strip()
    # question = input()
    print(f"Q: {question}")

    embedding = embedding_model.embed(question)
    distances = []
    
    for i in range(len(script_db)):
        chunk = script_db[i]
        distance = l2_distance(embedding, chunk['embedding'])
        distances.append((distance, chunk['script'], i))
    distances.sort(key=lambda x: x[0])
    
    near_chunks = distances[0:5]
    near_chunks.sort(key=lambda x: x[2])

    prompt = f"""Question: {question}
You can use these snippets of the script (they are in order):
# Snippet 1
{near_chunks[0][1]}
# Snippet 2
{near_chunks[1][1]}
# Snippet 3
{near_chunks[2][1]}
# Snippet 4
{near_chunks[3][1]}
# Snippet 5
{near_chunks[4][1]}
-----------------
Answer as succinctly as possible."""
    # print(prompt)
    chat = lms.Chat("You are a question-answer assistant for K-Pop Demon Hunter fans.")
    chat.add_user_message(prompt)
    result = llm_model.respond(chat)
    input_token_count += result.stats.prompt_tokens_count
    output_token_count += result.stats.predicted_tokens_count
    answer = result.content.strip()
    print(f"A: {answer}")

print(f"Input tokens: {input_token_count}")
print(f"Output tokens: {output_token_count}")