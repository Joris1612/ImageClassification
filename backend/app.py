from flask import Flask, request, jsonify
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
from io import BytesIO

from werkzeug.utils import secure_filename

from backend.predict import predict_single_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = 'path_to_your_model.h5'  # Update this path to your actual model


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # Predict using the saved image
        predicted_labels = predict_single_image(save_path)

        return jsonify({'predicted_labels': predicted_labels}), 200

if __name__ == '__main__':
    app.run(debug=True)