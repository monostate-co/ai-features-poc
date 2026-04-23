# index_images.py
# Generates CLIP embeddings from product images.
# Downloads each product image and encodes it with CLIP.

import csv
import json
import re
import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer

IMAGES_DIR = "product_images"
BATCH_SIZE = 32

def strip_html(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_products(csv_path):
    seen = {}
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
            color = row.get("Color (product.metafields.shopify.color-pattern)", "").strip()
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

def download_image(url, save_path):
    if os.path.exists(save_path):
        return True
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(save_path, "JPEG", quality=85)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False

def main():
    products = load_products("products_optimized.csv")
    print(f"Loaded {len(products)} unique products.")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    valid_products = []
    valid_image_paths = []

    print("Downloading images...")
    for i, p in enumerate(products):
        if not p["image"]:
            print(f"  [{i+1}/{len(products)}] Skipping {p['handle']} (no image URL)")
            continue

        save_path = os.path.join(IMAGES_DIR, f"{p['handle']}.jpg")
        success = download_image(p["image"], save_path)

        if success:
            valid_products.append(p)
            valid_image_paths.append(save_path)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(products)}] Downloaded {len(valid_products)} images so far...")

    print(f"\nSuccessfully downloaded {len(valid_products)} images out of {len(products)} products.")

    print("Loading CLIP model...")
    model = SentenceTransformer("clip-ViT-B-32")

    print("Generating image embeddings...")
    all_embeddings = []
    for i in range(0, len(valid_image_paths), BATCH_SIZE):
        batch_paths = valid_image_paths[i:i + BATCH_SIZE]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        batch_embeddings = model.encode(batch_images, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
        print(f"  Batch {i // BATCH_SIZE + 1}/{(len(valid_image_paths) - 1) // BATCH_SIZE + 1} done")

    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    np.save("embeddings_images.npy", embeddings)
    with open("products_images.json", "w", encoding="utf-8") as f:
        json.dump(valid_products, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(valid_products)} products. Saved embeddings_images.npy and products_images.json.")

if __name__ == "__main__":
    main()
