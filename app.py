from flask import Flask, request, jsonify
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import colorsys
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def generate_palette(image, palette_size=6, sample_size=10000):
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL.Image object")

    image = image.convert("RGB")
    r, g, b = np.array(image).reshape(-1, 3).T
    total_pixels = len(r)
    sample_size = min(sample_size, total_pixels)

    df = pd.DataFrame({"R": r, "G": g, "B": b}).sample(n=sample_size, replace=False)
    kmeans_model = KMeans(n_clusters=palette_size, random_state=1, init="k-means++", n_init="auto").fit(df)
    palette = kmeans_model.cluster_centers_.astype(int).tolist()
    palette.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb))
    list_hex = [rgb_to_hex(*rgb) for rgb in palette]
    return list_hex

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Get the desired palette size from request parameters
    palette_size = int(request.args.get('palette_size', 6))

    image = Image.open(io.BytesIO(file.read()))
    colors = generate_palette(image, palette_size)
    
    return jsonify(colors)

if __name__ == '__main__':
    app.run(debug=True)
