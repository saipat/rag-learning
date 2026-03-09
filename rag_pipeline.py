from openai import OpenAI
import math

client = OpenAI()

documents = [
    "Customers can return items within 30 days.",
    "Shipping usually takes 5 to 7 business days.",
    "Our office is located in San Francisco.",
    "You can reset your password from the settings page.",
    "We offer support through email and live chat.",
]


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text)
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2)


query = input("Ask a question: ")
query_embedding = get_embedding(query)

scores = []

for doc in documents:
    doc_embedding = get_embedding(doc)
    similarity = cosine_similarity(query_embedding, doc_embedding)
    scores.append((doc, similarity))

scores.sort(key=lambda x: x[1], reverse=True)

top_docs = scores[:3]

context = "\n".join([doc for doc, score in top_docs])

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}
"""

print(prompt)

response = client.chat.completions.create(
    model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
)

answer = response.choices[0].message.content

print("\nTop 3 matches:\n")
for doc, score in top_docs:
    print(f"{score:.4f} | {doc}")

print("\nFinal Answer:\n")
print(answer)
