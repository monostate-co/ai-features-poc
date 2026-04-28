# search.py
import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

embeddings = np.load("embeddings.npy")

with open("products.json", encoding="utf-8") as f:
    products = json.load(f)

# Build BM25 index from product text
def tokenize(text):
    return re.findall(r"\w+", text.lower())

bm25_corpus = [tokenize(p.get("text", "")) for p in products]
bm25_index = BM25Okapi(bm25_corpus)

def embed_query(text):
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding)

def search(query, top_k=5, bm25_weight=0.3):
    # Vector (semantic) scores
    query_vec = embed_query(query).reshape(1, -1)
    vec_scores = np.dot(embeddings, query_vec.T).squeeze()
    vec_scores /= (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec))

    # BM25 (keyword) scores
    query_tokens = tokenize(query)
    bm25_scores = np.array(bm25_index.get_scores(query_tokens))

    # Normalize both to [0, 1]
    vec_min, vec_max = vec_scores.min(), vec_scores.max()
    if vec_max > vec_min:
        vec_norm = (vec_scores - vec_min) / (vec_max - vec_min)
    else:
        vec_norm = np.zeros_like(vec_scores)

    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_norm = bm25_scores / bm25_max
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    # Combine: (1 - weight) * vector + weight * bm25
    combined = (1 - bm25_weight) * vec_norm + bm25_weight * bm25_norm
    top_indices = np.argsort(combined)[::-1][:top_k]

    results = []
    for i in top_indices:
        results.append({
    	    "title": products[i]["title"],
    	    "vendor": products[i]["vendor"],
    	    "score": float(combined[i]),
            "vector_score": float(vec_norm[i]),
            "vector_score_raw": float(vec_scores[i]),
            "bm25_score": float(bm25_norm[i]),
            "bm25_score_raw": float(bm25_scores[i]),
            "vector_weight": float(1 - bm25_weight),
            "bm25_weight": float(bm25_weight),
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
