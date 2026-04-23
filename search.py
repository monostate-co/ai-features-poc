# search.py
import json
import numpy as np
from openai import OpenAI

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

embeddings = np.load("embeddings.npy")

with open("products.json", encoding="utf-8") as f:
    products = json.load(f)

def embed_query(text):
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding)

def search(query, top_k=5):
    query_vec = embed_query(query).reshape(1, -1)
    scores = np.dot(embeddings, query_vec.T).squeeze()
    scores /= (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        results.append({
    	    "title": products[i]["title"],
    	    "vendor": products[i]["vendor"],
    	    "score": float(scores[i]),
            "handle": products[i]["handle"],
            "image": products[i].get("image", ""),
            "description": products[i].get("text", ""),
        })  
    return results

if __name__ == "__main__":
    while True:
        query = input("\nSearch: ")
        if query.lower() in ("quit", "exit"):
            break
        for r in search(query):
            print(f"  {r['score']:.3f}  {r['vendor']} — {r['title']}")
