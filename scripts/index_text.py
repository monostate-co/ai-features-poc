"""Incrementally sync the products_text Qdrant collection.

Reads `data/products.json`, computes a content hash per product, and
upserts only items that are new or whose hash changed. Points whose
handle no longer exists in the source are deleted.

Collection: one named dense vector ("dense", OpenAI text-embedding-3-small)
plus a sparse "bm25" vector with server-side IDF.

Run standalone:
  python scripts/index_text.py
"""

import hashlib
import json
import os
import uuid

from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models

PRODUCTS_PATH = "data/products.json"
COLLECTION_NAME = "products_text"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DENSE_BATCH = 500
SPARSE_MODEL = "Qdrant/bm25"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
UPSERT_BATCH = 256
SCROLL_BATCH = 1000

HASH_FIELDS = (
    "handle", "title", "vendor", "type", "category",
    "color", "image", "image_alt", "text",
)


def _bm25_text(p: dict) -> str:
    return " ".join(
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
    )


def _content_hash(p: dict) -> str:
    parts = [str(p.get(k, "")) for k in HASH_FIELDS]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _ensure_collection(qdrant: QdrantClient) -> None:
    if qdrant.collection_exists(COLLECTION_NAME):
        return
    print(f"Creating collection '{COLLECTION_NAME}'...")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


def _existing_hashes(qdrant: QdrantClient) -> dict[str, str]:
    out: dict[str, str] = {}
    next_offset = None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_BATCH,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            out[str(pt.id)] = (pt.payload or {}).get("_hash", "")
        if next_offset is None:
            break
    return out


def _dense_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    n_batches = (len(texts) - 1) // DENSE_BATCH + 1
    for i in range(0, len(texts), DENSE_BATCH):
        batch = texts[i : i + DENSE_BATCH]
        print(f"  dense batch {i // DENSE_BATCH + 1}/{n_batches}...")
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        out.extend(e.embedding for e in resp.data)
    return out


def sync() -> None:
    with open(PRODUCTS_PATH, encoding="utf-8") as f:
        products = json.load(f)
    products = [p for p in products if p.get("image_alt")]

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    _ensure_collection(qdrant)

    desired: dict[str, tuple[dict, str]] = {}
    for p in products:
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, p["handle"]))
        desired[pid] = (p, _content_hash(p))

    existing = _existing_hashes(qdrant)

    upsert_pids = [pid for pid, (_, h) in desired.items() if existing.get(pid) != h]
    delete_pids = [pid for pid in existing if pid not in desired]
    unchanged = len(desired) - len(upsert_pids)
    print(
        f"text sync: {len(upsert_pids)} upsert, {len(delete_pids)} delete, "
        f"{unchanged} unchanged"
    )

    if upsert_pids:
        upsert_products = [desired[pid][0] for pid in upsert_pids]
        upsert_hashes = [desired[pid][1] for pid in upsert_pids]
        dense_texts = [p["image_alt"] for p in upsert_products]
        bm25_texts = [_bm25_text(p) for p in upsert_products]

        print("Encoding dense vectors with OpenAI...")
        openai_client = OpenAI()
        dense_vecs = _dense_embeddings(openai_client, dense_texts)

        print(f"Encoding sparse vectors with {SPARSE_MODEL}...")
        sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        # parallel=None encodes inline; parallel>=1 spawns a forkserver worker
        # that re-imports app.py and re-enters sync() during bootstrap.
        sparse_vecs = list(sparse_model.embed(bm25_texts, parallel=None, batch_size=64))

        points = []
        for pid, p, h, dense, sparse in zip(
            upsert_pids, upsert_products, upsert_hashes, dense_vecs, sparse_vecs
        ):
            payload = dict(p)
            payload["_hash"] = h
            points.append(
                models.PointStruct(
                    id=pid,
                    vector={
                        "dense": dense,
                        "bm25": models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

        print(f"Upserting {len(points)} points in batches of {UPSERT_BATCH}...")
        for i in range(0, len(points), UPSERT_BATCH):
            batch = points[i : i + UPSERT_BATCH]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
            print(f"  upserted {min(i + UPSERT_BATCH, len(points))}/{len(points)}")

    if delete_pids:
        print(f"Deleting {len(delete_pids)} stale points...")
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=delete_pids),
            wait=True,
        )

    print("text sync done.")


if __name__ == "__main__":
    sync()
