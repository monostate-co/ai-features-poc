"""Re-create text embeddings and upsert them to Qdrant.

Builds one collection with two named vectors:
  - "dense": OpenAI text-embedding-3-small (1536 dims, cosine)
  - "bm25":  FastEmbed Qdrant/bm25 sparse vectors with server-side IDF

Run:
  python scripts/index_text.py
"""

import csv
import os
import re
import uuid

from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models

CSV_PATH = "data/products.csv"
COLLECTION_NAME = "products_text"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DENSE_BATCH = 500
SPARSE_MODEL = "Qdrant/bm25"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_products(csv_path: str) -> list[dict]:
    seen: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            handle = row.get("Handle", "").strip()
            if not handle or handle in seen:
                continue

            title = row.get("Title", "").strip()
            body = strip_html(row.get("Body (HTML)", ""))
            vendor = row.get("Vendor", "").strip()
            ptype = row.get("Type", "").strip()
            tags = row.get("Tags", "").strip()
            image = row.get("Image Src", "").strip()
            category = row.get("Product Category", "").strip()
            color = row.get(
                "Color (product.metafields.shopify.color-pattern)", ""
            ).strip()
            image_alt = row.get("Image Alt Text", "").strip()

            parts = [title, body]
            if vendor:
                parts.append(f"Brand: {vendor}")
            if category:
                parts.append(f"Category: {category}")

            seen[handle] = {
                "handle": handle,
                "title": title,
                "vendor": vendor,
                "type": ptype,
                "tags": tags,
                "category": category,
                "color": color,
                "image": image,
                "image_alt": image_alt,
                "text": ". ".join(p for p in parts if p),
            }

    return list(seen.values())


def dense_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), DENSE_BATCH):
        batch = texts[i : i + DENSE_BATCH]
        n_batches = (len(texts) - 1) // DENSE_BATCH + 1
        print(f"  dense batch {i // DENSE_BATCH + 1}/{n_batches}...")
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        out.extend(e.embedding for e in resp.data)
    return out


def ensure_collection(qdrant: QdrantClient) -> None:
    if qdrant.collection_exists(COLLECTION_NAME):
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        qdrant.delete_collection(COLLECTION_NAME)
    else:
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


def main() -> None:
    products = load_products(CSV_PATH)
    products = [p for p in products if p["image_alt"]]
    print(f"Loaded {len(products)} products with image_alt.")

    # Text used for both dense and sparse — keep them aligned so BM25
    # and semantic search agree on what each product "is."
    bm25_texts = [
        " ".join(
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
        for p in products
    ]
    dense_texts = [p["image_alt"] for p in products]

    print("Encoding dense vectors with OpenAI...")
    openai_client = OpenAI()
    dense_vecs = dense_embeddings(openai_client, dense_texts)

    print(f"Encoding sparse vectors with {SPARSE_MODEL}...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    sparse_vecs = list(sparse_model.embed(bm25_texts, parallel=1, batch_size=64))

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant)

    points = []
    for product, dense, sparse in zip(products, dense_vecs, sparse_vecs):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, product["handle"])),
                vector={
                    "dense": dense,
                    "bm25": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload=product,
            )
        )

    UPSERT_BATCH = 256
    print(f"Upserting {len(points)} points in batches of {UPSERT_BATCH}...")
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i : i + UPSERT_BATCH]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        print(f"  upserted {min(i + UPSERT_BATCH, len(points))}/{len(points)}")
    print("Done.")


if __name__ == "__main__":
    main()
