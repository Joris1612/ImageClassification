from flask import Flask, jsonify, request
import base64
import os

app = Flask(__name__)


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


@app.route('/')
def hello_world():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)
