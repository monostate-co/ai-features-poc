# search_images.py
# Image-to-image search using CLIP embeddings.

import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("clip-ViT-B-32")

embeddings = np.load("data/embeddings_images.npy")

with open("data/products_images.json", encoding="utf-8") as f:
    products = json.load(f)

def search_by_image(image, top_k=10):
    """Search products by image similarity. Accepts a PIL Image or file path."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    query_vec = model.encode([image])[0].reshape(1, -1)
    scores = np.dot(embeddings, query_vec.T).squeeze()
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    scores = scores / norms
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
