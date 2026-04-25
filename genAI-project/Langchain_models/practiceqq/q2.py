from openai import OpenAI
import numpy as np
 
# Initialize OpenAI client
client = OpenAI(api_key="your_api_key_here")
 
# -----------------------------
# 1. Get embedding for text
# -----------------------------
def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    # Correct key is 'embedding' (not embeddings)
    return np.array(res.data[0].embedding, dtype='float32')
 
 
# -----------------------------
# 2. Cosine similarity function
# -----------------------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
 
 
# -----------------------------
# 3. Documents (your data)
# -----------------------------
documents = [
    "mera naam yashwant hai",
    "meri age 21 hai"
]
 
# Create embeddings for all documents
doc_emb = [get_embedding(doc) for doc in documents]
 
 
# -----------------------------
# 4. Search function
# -----------------------------
def search(query):
    query_emb = get_embedding(query)
 
    # Calculate similarity score for each document
    results = [
        {
            "text": doc,
            "score": cosine(query_emb, emb)
        }
        for doc, emb in zip(documents, doc_emb)
    ]
 
    # Sort results by highest similarity score
    return sorted(results, key=lambda x: x["score"], reverse=True)
 
 
# -----------------------------
# 5. Run search
# -----------------------------
query = input("Enter your query: ")
 
results = search(query)
 
print("\nTop results:\n")
 
for r in results:
    print(f"{r['text']} → {r['score']:.3f}")