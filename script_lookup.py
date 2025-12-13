import json
import math
import random
import lmstudio as lms

model = lms.embedding_model("text-embedding-qwen3-embedding-0.6b")

def l2_distance(a, b):
    sum = 0
    for i in range(0, len(a)):
        difference = a[i] - b[i]
        sum += difference ** 2
    return math.sqrt(sum)

#with open("reviews_embedding.json", "r") as f:
#    reviews = json.load(f)

with open("script_db.json", "r") as f:
    script_db = json.load(f)

print("Enter something to find in the script")
query = input()
query_embedding = model.embed(query)

nearest = None
min_distance = None
closest = None
for chunk in script_db:
    distance = l2_distance(query_embedding, chunk['embedding'])
    if min_distance is None:
        min_distance = distance
        closest = chunk['script']
    elif distance < min_distance:
        min_distance = distance
        closest = chunk['script']
print(closest)