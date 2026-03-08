from openai import OpenAI
import math

client = OpenAI()

text1 = "Return policy"
text2 = "Refund policy"

response1 = client.embeddings.create(
    model="text-embedding-3-small",
    input=text1)
response2 = client.embeddings.create(
    model="text-embedding-3-small",
    input=text2)

embedding1 = response1.data[0].embedding
embedding2 = response2.data[0].embedding

# Cosine similarity measures the angle between vectors


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    # hypotenuse² = a² + b² - length of the vector
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2)


similarity = cosine_similarity(embedding1, embedding2)

print("Embedding length:", len(embedding1))
print("Embedding length:", len(embedding2))
print("Cosine similarity:", similarity)
