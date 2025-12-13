import lmstudio as lms
import os
import json

model = lms.embedding_model("text-embedding-qwen3-embedding-0.6b")

concept_1 = "K-Pop"
concept_2 = "Country Music"
embedding_1 = model.embed(concept_1)
embedding_2 = model.embed(concept_2)

# Just print the first 5 elements of the embedding ... it's 768 elements in total
print(f"{concept_1}: {embedding_1[0:5]}...")
print(f"{concept_2}: {embedding_2[0:5]}...")

def cosine_similarity(vec_a, vec_b):
    # 1. Compute the dot product
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    
    # 2. Compute the magnitudes (Euclidean norms)
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5
    
    # 3. Prevent division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
        
    # 4. Return cosine similarity
    return dot_product / (magnitude_a * magnitude_b)

def l2_distance(vec_a, vec_b):
    # Sum of squared differences
    sum_sq_diff = sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))
    
    # Return the square root
    return sum_sq_diff ** 0.5

print(f"cosine similairty: {cosine_similarity(embedding_1, embedding_2)}")
print(f"euclidean distance: {l2_distance(embedding_1, embedding_2)}")