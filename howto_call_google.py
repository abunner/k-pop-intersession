# First do this in your terminal
# uv env
# source .venv/bin/activate 
# uv pip install google-genai

# Call this by doing this on the command line:
# API_KEY=<SECRET> python howto_call_google.py

# ... but you have to get the secret from the instructor

from google import genai
from google.genai.types import HttpOptions
import os

client = genai.Client(
    http_options=HttpOptions(api_version="v1", timeout=600_000),
    vertexai=True,
    api_key=os.environ.get("APIK_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents="What's the capital of France?"
)
print(response.text)
print(f"Input token count: {response.usage_metadata.prompt_token_count}")
print(f"Output token count: {response.usage_metadata.candidates_token_count}")

concept_1 = "K-Pop"
concept_2 = "Country Music"
response_1 = client.models.embed_content(model="gemini-embedding-001", contents=concept_1)
response_2 = client.models.embed_content(model="gemini-embedding-001", contents=concept_2)
embedding_1 = response_1.embeddings[0].values
embedding_2 = response_2.embeddings[0].values

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