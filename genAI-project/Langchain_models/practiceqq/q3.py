from openai import OpenAI
import numpy as np
 
client = OpenAI(api_key="your_api_key_here")
 
# -----------------------------
# 1. Load documents from file
# -----------------------------
def load_documents(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]
 
 
documents = load_documents("data.txt")   # 👈 external data
 
 
# -----------------------------
# 2. Get embedding
# -----------------------------
def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding, dtype='float32')
 
 
# -----------------------------
# 3. Cosine similarity
# -----------------------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
 
 
# -----------------------------
# 4. Create embeddings
# -----------------------------
doc_emb = [get_embedding(doc) for doc in documents]
 
 
# -----------------------------
# 5. Search (Top-K)
# -----------------------------
def search(query, k=2):
    query_emb = get_embedding(query)
 
    results = [
        {"text": doc, "score": cosine(query_emb, emb)}
        for doc, emb in zip(documents, doc_emb)
    ]
 
    results = sorted(results, key=lambda x: x["score"], reverse=True)
 
    return results[:k]
 
 
# -----------------------------
# 6. Generate answer
# -----------------------------
def generate_answer(query, context_docs):
    context = "\n".join(doc["text"] for doc in context_docs)
 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Answer using only this context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )
 
    return response.choices[0].message.content
 
 
# -----------------------------
# 7. Run
# -----------------------------
query = input("Enter your query: ")
 
top_docs = search(query, k=2)
 
print("\nTop Retrieved Docs:\n")
for d in top_docs:
    print(f"{d['text']} → {d['score']:.3f}")
 
answer = generate_answer(query, top_docs)
 
print("\nFinal Answer:\n")
print(answer)