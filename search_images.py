"""Image-to-image search backed by Qdrant.

Encodes the query image with CLIP and queries the products_images collection.
"""

import os

from PIL import Image
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "products_images"
MODEL_NAME = "clip-ViT-B-32"
MODEL_PATH = os.path.join("models", MODEL_NAME)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _load_model() -> SentenceTransformer:
    if os.path.exists(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH)
    model = SentenceTransformer(MODEL_NAME)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    return model


model = _load_model()
qdrant = QdrantClient(url=QDRANT_URL)


def search_by_image(image, top_k: int = 10):
    """Search products by image similarity. Accepts a PIL Image or file path."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    query_vec = model.encode([image])[0].tolist()

    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="image",
        limit=top_k,
        with_payload=True,
    ).points

    results = []
    for hit in hits:
        p = hit.payload or {}
        results.append(
            {
                "title": p.get("title", ""),
                "vendor": p.get("vendor", ""),
                "score": float(hit.score),
                "handle": p.get("handle", ""),
                "image": p.get("image", ""),
                "description": p.get("text", ""),
            }
        )
    return results
