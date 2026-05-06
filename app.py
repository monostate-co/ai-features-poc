import os
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from io import BytesIO

from scripts.index_text import sync as sync_text
from scripts.index_images import sync as sync_images
from search_text import search
from search_images import search_by_image

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.route("/")
def index():
    return send_file("templates/index.html")

@app.route("/search")
def api_search():
    q = request.args.get("q", "")
    results = search(q, top_k=24)
    return jsonify(results)

@app.route("/search-by-image", methods=["POST"])
def api_search_by_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(BytesIO(file.read())).convert("RGB")
    results = search_by_image(image, top_k=10)
    return jsonify(results)

def _background_sync():
    print("Starting Qdrant sync...")
    sync_text()
    sync_images()
    print("Qdrant sync complete.")


if __name__ == "__main__":
    # Sync inside the __main__ guard so a forkserver/spawn worker re-importing
    # this module to bootstrap doesn't re-enter sync() mid-spawn. Run it in a
    # daemon thread so Flask binds the port immediately — otherwise Coolify's
    # proxy returns 502 for the entire sync window.
    threading.Thread(target=_background_sync, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
