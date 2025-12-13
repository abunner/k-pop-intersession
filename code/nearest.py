import json
import math
import lmstudio as lms

def l2_distance(a, b):
    sum = 0
    for i in range(0, len(a)):
        difference = a[i] - b[i]
        sum += difference ** 2
    return math.sqrt(sum)


with open("reviews_embedding.json", "r") as f:
    data = json.load(f)

model = lms.embedding_model("text-embedding-qwen3-embedding-0.6b")
target = model.embed("Soja boys are hot")

min_distance = None
closest = None
for review_embedding in data:
    distance = l2_distance(target, review_embedding['embedding'])
    if min_distance is None:
        min_distance = distance
        closest = review_embedding['review']
    elif distance < min_distance:
        min_distance = distance
        closest = review_embedding['review']

print(closest)
print(min_distance)