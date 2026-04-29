# index.py
import csv
import json
import re
import numpy as np
from openai import OpenAI

def strip_html(text):
    """Remove HTML tags and clean up whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_products(csv_path):
    """Load products from Shopify CSV, one entry per handle."""
    seen = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            handle = row.get("Handle", "").strip()
            if not handle:
                continue
            # Only take the first row per handle (has the title/description)
            if handle in seen:
                continue

            title = row.get("Title", "").strip()
            body = strip_html(row.get("Body (HTML)", ""))
            vendor = row.get("Vendor", "").strip()
            ptype = row.get("Type", "").strip()
            tags = row.get("Tags", "").strip()
            image = row.get("Image Src", "").strip()
            category = row.get("Product Category", "").strip()
            seo_desc = row.get("SEO Description", "").strip()
            color = row.get("Color (product.metafields.shopify.color-pattern)", "").strip()
            image_alt = row.get("Image Alt Text", "").strip()
            complementary = row.get("Complementary products (product.metafields.shopify--discovery--product_recommendation.complementary_products)", "").strip()

            # Build a combined text block for embedding
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
                "complementary": complementary,
                "text": ". ".join(p for p in parts if p),
            }

    return list(seen.values())

# 1. Load products
products = load_products("data/products_optimized.csv")
print(f"Loaded {len(products)} unique products.")

# 2. Encode with OpenAI
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

def get_embeddings(texts, batch_size=500):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Encoding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        all_embeddings.extend([e.embedding for e in response.data])
    return np.array(all_embeddings)

texts = [p["image_alt"] for p in products]
embeddings = get_embeddings(texts)

# 4. Save
np.save("data/embeddings.npy", embeddings)
with open("data/products.json", "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False, indent=2)

print(f"Indexed {len(products)} products. Saved embeddings.npy and products.json.")
