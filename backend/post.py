import os
import base64
import requests
import json


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def send_image(url, encoded_image):
    """Send the encoded image to the server."""
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'image': encoded_image})

    response = requests.post(url, headers=headers, data=payload)
    return response.json()


# Define the image directory and filename
image_directory = '../images'
image_filename = '00000001_000.png'
image_path = os.path.join(image_directory, image_filename)

# Encode the image
encoded_image = encode_image(image_path)

# URL to your Flask endpoint
url = 'http://localhost:5000/api/predict'

# Send the image and print the response
response = send_image(url, encoded_image)
print(response)
