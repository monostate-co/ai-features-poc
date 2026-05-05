"""Re-create CLIP image embeddings and upsert them to Qdrant.

Builds a single-vector collection:
  - "image": CLIP ViT-B-32 (512 dims, cosine)

Run:
  python scripts/index_images.py
"""

import csv
import os
import re
import uuid

import requests
from PIL import Image
from io import BytesIO
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/products.csv"
COLLECTION_NAME = "products_images"
IMAGES_DIR = "product_images"
MODEL_NAME = "clip-ViT-B-32"
MODEL_PATH = os.path.join("models", MODEL_NAME)
EMBEDDING_DIM = 512
ENCODE_BATCH = 32
UPSERT_BATCH = 256
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
            category = row.get("Product Category", "").strip()
            color = row.get(
                "Color (product.metafields.shopify.color-pattern)", ""
            ).strip()
            image = row.get("Image Src", "").strip()
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
                "category": category,
                "color": color,
                "image": image,
                "image_alt": image_alt,
                "text": ". ".join(p for p in parts if p),
            }
    return list(seen.values())


def download_image(url: str, save_path: str) -> bool:
    if os.path.exists(save_path):
        return True
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        Image.open(BytesIO(resp.content)).convert("RGB").save(
            save_path, "JPEG", quality=85
        )
        return True
    except Exception as e:
        print(f"  failed to download {url}: {e}")
        return False


def load_clip_model() -> SentenceTransformer:
    if os.path.exists(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH)
    model = SentenceTransformer(MODEL_NAME)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    return model


def ensure_collection(qdrant: QdrantClient) -> None:
    if qdrant.collection_exists(COLLECTION_NAME):
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        qdrant.delete_collection(COLLECTION_NAME)
    else:
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


def main() -> None:
    products = load_products(CSV_PATH)
    print(f"Loaded {len(products)} unique products.")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    valid_products: list[dict] = []
    valid_image_paths: list[str] = []

    print("Resolving images (downloading any that aren't cached)...")
    for i, p in enumerate(products):
        if not p["image"]:
            continue
        save_path = os.path.join(IMAGES_DIR, f"{p['handle']}.jpg")
        if download_image(p["image"], save_path):
            valid_products.append(p)
            valid_image_paths.append(save_path)
        if (i + 1) % 250 == 0:
            print(f"  [{i + 1}/{len(products)}] resolved {len(valid_products)} so far...")
    print(f"Have {len(valid_products)} usable images.")

    print("Loading CLIP model...")
    model = load_clip_model()

    print("Generating image embeddings...")
    all_embeddings = []
    for i in range(0, len(valid_image_paths), ENCODE_BATCH):
        batch_paths = valid_image_paths[i : i + ENCODE_BATCH]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        batch_emb = model.encode(batch_images, show_progress_bar=False)
        all_embeddings.extend(batch_emb)
        n_batches = (len(valid_image_paths) - 1) // ENCODE_BATCH + 1
        print(f"  encode batch {i // ENCODE_BATCH + 1}/{n_batches} done")

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant)

    points = []
    for product, vec in zip(valid_products, all_embeddings):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, product["handle"])),
                vector={"image": vec.tolist()},
                payload=product,
            )
        )

    print(f"Upserting {len(points)} points in batches of {UPSERT_BATCH}...")
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i : i + UPSERT_BATCH]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        print(f"  upserted {min(i + UPSERT_BATCH, len(points))}/{len(points)}")
    print("Done.")


if __name__ == "__main__":
    main()
