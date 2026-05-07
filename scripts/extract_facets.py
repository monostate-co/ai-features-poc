"""Extract metadata facets from a free-text query.

Builds a vocabulary of vendor / color / type / category-segment values
from the products dataset, then scans queries for case-insensitive
whole-word matches. Longer values are tried first so multi-word vendors
like "Jil sander" win over any prefix.

Run standalone to probe interactively:
  python scripts/extract_facets.py
"""

import json
import re
from functools import lru_cache

PRODUCTS_PATH = "data/products_images.json"
SIMPLE_FIELDS = ("vendor", "color", "type")


@lru_cache(maxsize=1)
def _vocabulary() -> dict[str, list[str]]:
    with open(PRODUCTS_PATH, encoding="utf-8") as f:
        products = json.load(f)

    vocab: dict[str, set[str]] = {f: set() for f in (*SIMPLE_FIELDS, "category")}
    for p in products:
        for field in SIMPLE_FIELDS:
            v = p.get(field)
            if v:
                vocab[field].add(v)
        # Category is a path like "Apparel & Accessories > Clothing > Outerwear > Vests".
        # Split into segments so "outerwear" or "vests" can match.
        for seg in (p.get("category") or "").split(" > "):
            seg = seg.strip()
            if seg:
                vocab["category"].add(seg)

    return {f: sorted(v, key=len, reverse=True) for f, v in vocab.items()}


def extract(query: str) -> dict[str, list[str]]:
    """Return a {field: [matched values]} dict for facets found in `query`."""
    if not query:
        return {}
    vocab = _vocabulary()
    found: dict[str, list[str]] = {}
    consumed: list[tuple[int, int]] = []
    lower = query.lower()

    for field in (*SIMPLE_FIELDS, "category"):
        for value in vocab[field]:
            pattern = r"\b" + re.escape(value.lower()) + r"\b"
            for m in re.finditer(pattern, lower):
                span = (m.start(), m.end())
                # Skip if this span overlaps something a longer/earlier match took.
                if any(s < span[1] and span[0] < e for s, e in consumed):
                    continue
                consumed.append(span)
                found.setdefault(field, []).append(value)

    return found


if __name__ == "__main__":
    while True:
        try:
            q = input("\nquery: ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit"):
            break
        print(extract(q))
