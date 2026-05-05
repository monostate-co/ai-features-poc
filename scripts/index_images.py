"""Incrementally sync the products_images Qdrant collection.

Reads `data/products_images.json`, ensures each handle has a cached
image in `product_images/`, and upserts CLIP image embeddings only for
products that are new or whose payload hash changed. Stale points and
local image files are removed.

Run standalone:
  python scripts/index_images.py
"""

import hashlib
import json
import os
import uuid
from io import BytesIO

import requests
from PIL import Image
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

PRODUCTS_PATH = "data/products_images.json"
COLLECTION_NAME = "products_images"
IMAGES_DIR = "product_images"
MODEL_NAME = "clip-ViT-B-32"
MODEL_PATH = os.path.join("models", MODEL_NAME)
EMBEDDING_DIM = 512
ENCODE_BATCH = 32
UPSERT_BATCH = 256
SCROLL_BATCH = 1000
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None

HASH_FIELDS = (
    "handle", "title", "vendor", "type", "category",
    "color", "image", "image_alt", "text",
)


def _content_hash(p: dict) -> str:
    parts = [str(p.get(k, "")) for k in HASH_FIELDS]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _download_image(url: str, save_path: str) -> bool:
    if os.path.exists(save_path):
        return True
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        Image.open(BytesIO(resp.content)).convert("RGB").save(
            save_path, "JPEG", quality=85
        )
        return True
    except Exception as e:
        print(f"  failed to download {url}: {e}")
        return False


def _load_clip_model() -> SentenceTransformer:
    if os.path.exists(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH)
    model = SentenceTransformer(MODEL_NAME)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    return model


def _ensure_collection(qdrant: QdrantClient) -> None:
    if qdrant.collection_exists(COLLECTION_NAME):
        return
    print(f"Creating collection '{COLLECTION_NAME}'...")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "image": models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
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


def _prune_local_images(desired_handles: set[str]) -> int:
    if not os.path.isdir(IMAGES_DIR):
        return 0
    removed = 0
    for fn in os.listdir(IMAGES_DIR):
        if fn.endswith(".jpg") and fn[:-4] not in desired_handles:
            os.remove(os.path.join(IMAGES_DIR, fn))
            removed += 1
    return removed


def sync() -> None:
    with open(PRODUCTS_PATH, encoding="utf-8") as f:
        products = json.load(f)
    products = [p for p in products if p.get("image")]

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
        f"image sync: {len(upsert_pids)} upsert, {len(delete_pids)} delete, "
        f"{unchanged} unchanged"
    )

    if upsert_pids:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        valid_pids: list[str] = []
        valid_products: list[dict] = []
        valid_hashes: list[str] = []
        valid_paths: list[str] = []
        for pid in upsert_pids:
            p, h = desired[pid]
            save_path = os.path.join(IMAGES_DIR, f"{p['handle']}.jpg")
            if not _download_image(p["image"], save_path):
                continue
            valid_pids.append(pid)
            valid_products.append(p)
            valid_hashes.append(h)
            valid_paths.append(save_path)

        print("Loading CLIP model...")
        model = _load_clip_model()

        print(f"Encoding {len(valid_paths)} images...")
        embeddings = []
        n_batches = (len(valid_paths) - 1) // ENCODE_BATCH + 1
        for i in range(0, len(valid_paths), ENCODE_BATCH):
            batch_paths = valid_paths[i : i + ENCODE_BATCH]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            batch_emb = model.encode(batch_images, show_progress_bar=False)
            embeddings.extend(batch_emb)
            print(f"  encode batch {i // ENCODE_BATCH + 1}/{n_batches} done")

        points = []
        for pid, p, h, vec in zip(valid_pids, valid_products, valid_hashes, embeddings):
            payload = dict(p)
            payload["_hash"] = h
            points.append(
                models.PointStruct(
                    id=pid,
                    vector={"image": vec.tolist()},
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

    pruned = _prune_local_images({p["handle"] for p in products})
    if pruned:
        print(f"Removed {pruned} stale local images.")

    print("image sync done.")


if __name__ == "__main__":
    sync()
