import os

import tensorflow as tf
import numpy as np
from PIL import Image


def predict_single_image( img_path):
    # Load the model
    model_path = '../XRAY-E10-multi-label-v2.keras'
    class_names = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]
    img_size = (128, 128)
    model = tf.keras.models.load_model(model_path)
    print(f'predicting image {img_path}')
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
    predicted_percentages = [float(pred * 100) for pred in predictions]

    predicted_labels = dict(zip(class_names, predicted_percentages))

    return predicted_labels
