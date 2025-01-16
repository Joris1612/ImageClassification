from flask import Flask, jsonify, request
import base64
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your model (adjust path as needed)
model = tf.keras.models.load_model('../XRAY-E10-multi-label.keras')
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]


def prepare_image(image_data, target_size=(128, 128)):
    """Convert base64 image data to a numpy array suitable for model input."""
    image = Image.open(io.BytesIO(image_data))
    image = image.resize(target_size)
    image_array = np.array(image)

    # Handling grayscale or single-channel images
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 1:
        image_array = np.repeat(image_array, 3, axis=2)

    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)


@app.route('/api/upload', methods=['POST'])
def upload_image():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_data = data['image']
    try:
        image_bytes = base64.b64decode(image_data)
        with open(os.path.join('uploads', 'uploaded_image.jpg'), 'wb') as f:
            f.write(image_bytes)
        return jsonify({'message': 'Image successfully uploaded'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_image():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_data = data['image']
    try:
        image_bytes = base64.b64decode(image_data)
        img_array = prepare_image(image_bytes)
        predictions = model.predict(img_array)[0]
        predicted_labels = [class_names[i] for i, prob in enumerate(predictions) if prob > 0.5]

        return jsonify({'predictions': predicted_labels}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def hello_world():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)
