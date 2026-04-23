from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from search import search
from search_images import search_by_image

app = Flask(__name__)
CORS(app)

@app.route("/search")
def api_search():
    q = request.args.get("q", "")
    results = search(q, top_k=10)
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
    app.run(port=5000)
