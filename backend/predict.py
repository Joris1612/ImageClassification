import os

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def predict_single_image(model_path, img_path, class_names, threshold=0.5, img_size=(128, 128)):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img = Image.open(img_path).resize(img_size)  # Use PIL to open and resize image
    img_array = np.array(img) / 255.0  # Convert image to numpy array and normalize

    # Check if the image is grayscale and convert it to RGB if needed
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 1:  # For some formats, image might be loaded with a single channel
        img_array = np.repeat(img_array, 3, axis=2)

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)[0]  # Get the first and only batch

    # Convert probabilities to binary labels
    predicted_labels = [class_names[i] for i, prob in enumerate(predictions) if prob > threshold]

    # Plotting the image
    plt.imshow(img.convert('RGB'))  # Ensure image is displayed in RGB format
    plt.title(f"Predicted: {', '.join(predicted_labels)}")
    plt.axis('off')
    plt.show()

    return predictions, predicted_labels


# Example usage:
model_path = '../XRAY-E10-multi-label.keras'  # Update path as necessary
image_path = os.path.join('../images', '00000001_001.png')  # Update to your image path
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

predictions, predicted_labels = predict_single_image(model_path, image_path, class_names)
