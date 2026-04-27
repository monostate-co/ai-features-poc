import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from search import search
from search_images import search_by_image

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_file("frontend.html")

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
