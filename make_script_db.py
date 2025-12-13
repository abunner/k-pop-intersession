import lmstudio as lms
import os
import json

model = lms.embedding_model("text-embedding-qwen3-embedding-0.6b")

with open("script.txt", "r") as f:
    lines = f.readlines()

chunks = []
i = 0
while i < len(lines):
    chunk = lines[i]
    i += 1
    while i < len(lines) and len(chunk) + len(lines[i]) < 1024:
        chunk += lines[i]
        i += 1
    print("CHUNKS")
    print(chunk)
    chunks.append(chunk)

data = []
for chunk in chunks:
    embedding = model.embed(chunk)
    data.append({ "script": chunk, "embedding": embedding})

with open("script_db.json", "w") as f:
    json.dump(data, f)