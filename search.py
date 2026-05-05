"""Hybrid product search backed by Qdrant.

Runs two queries against the same collection — one over the dense
(OpenAI) vector and one over the BM25 sparse vector — then merges the
results so the UI can still reweight signals client-side.
"""

import os

import numpy as np
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "products_text"
EMBEDDING_MODEL = "text-embedding-3-small"
SPARSE_MODEL = "Qdrant/bm25"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None

# Pull a wider candidate set from each branch than we ultimately return,
# so the merged top-k reflects items that score well on either signal.
CANDIDATE_LIMIT = 100

openai_client = OpenAI()
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)


def embed_query(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def sparse_query(text: str) -> models.SparseVector:
    sparse = next(iter(sparse_model.query_embed(text)))
    return models.SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )


def search(query: str, top_k: int = 5, bm25_weight: float = 0.3):
    dense_hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_query(query),
        using="dense",
        limit=CANDIDATE_LIMIT,
        with_payload=True,
    ).points

    sparse_hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_query(query),
        using="bm25",
        limit=CANDIDATE_LIMIT,
        with_payload=True,
    ).points

    merged: dict[str, dict] = {}
    for hit in dense_hits:
        merged[hit.id] = {
            "payload": hit.payload,
            "vector_score_raw": float(hit.score),
            "bm25_score_raw": 0.0,
        }
    for hit in sparse_hits:
        entry = merged.setdefault(
            hit.id,
            {
                "payload": hit.payload,
                "vector_score_raw": 0.0,
                "bm25_score_raw": 0.0,
            },
        )
        entry["bm25_score_raw"] = float(hit.score)

    if not merged:
        return []

    vec_raw = np.array([m["vector_score_raw"] for m in merged.values()])
    bm25_raw = np.array([m["bm25_score_raw"] for m in merged.values()])

    vec_min, vec_max = vec_raw.min(), vec_raw.max()
    if vec_max > vec_min:
        vec_norm = (vec_raw - vec_min) / (vec_max - vec_min)
    else:
        vec_norm = np.zeros_like(vec_raw)

    bm25_max = bm25_raw.max()
    if bm25_max > 0:
        bm25_norm = bm25_raw / bm25_max
    else:
        bm25_norm = np.zeros_like(bm25_raw)

    combined = (1 - bm25_weight) * vec_norm + bm25_weight * bm25_norm
    order = np.argsort(combined)[::-1][:top_k]

    entries = list(merged.values())
    results = []
    for i in order:
        entry = entries[i]
        p = entry["payload"]
        results.append(
            {
                "title": p.get("title", ""),
                "vendor": p.get("vendor", ""),
                "score": float(combined[i]),
                "vector_score": float(vec_norm[i]),
                "vector_score_raw": float(vec_raw[i]),
                "bm25_score": float(bm25_norm[i]),
                "bm25_score_raw": float(bm25_raw[i]),
                "vector_weight": float(1 - bm25_weight),
                "bm25_weight": float(bm25_weight),
                "handle": p.get("handle", ""),
                "image": p.get("image", ""),
                "description": p.get("text", ""),
                "bm25_text": " ".join(
                    filter(
                        None,
                        [
                            p.get("image_alt", ""),
                            p.get("vendor", ""),
                            p.get("type", ""),
                            p.get("color", ""),
                            p.get("text", ""),
                        ],
                    )
                ),
            }
        )
    return results


if __name__ == "__main__":
    while True:
        q = input("\nSearch: ")
        if q.lower() in ("quit", "exit"):
            break
        for r in search(q):
            print(f"  {r['score']:.3f}  {r['vendor']} — {r['title']}")
