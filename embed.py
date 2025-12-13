import lmstudio as lms
import os
import json

model = lms.embedding_model("text-embedding-qwen3-embedding-0.6b")

with open("reviews.txt", "r") as f:
    lines = f.readlines()

embedding = model.embed("hello world")
data = []

with open("reviews_embedding.json", "w") as f:
    for line in lines:
        embedding = model.embed(line)
        data.append({
            "review": line,
            "embedding": embedding 
        })
    json.dump(data, f)