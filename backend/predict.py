import os

import tensorflow as tf
import numpy as np
from PIL import Image


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = tf.expand_dims(img, axis=0)
    return img


def predict_single_image(img_path):
    # Load the model
    model_path = '../XRAY-E10-multi-label.keras'
    # model_path = "../XRAY-model-E100.keras"
    class_names = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    model = tf.keras.models.load_model(model_path)
    print(f'predicting image {img_path}')
    # Load and preprocess the image
    img = process_path(img_path)
    predictions = model.predict(img)[0]  # Get the first and only batch
    predicted_percentages = [float(pred * 100) for pred in predictions]

    predicted_labels = dict(zip(class_names, predicted_percentages))

    return predicted_labels


if __name__ == '__main__':
    predict_1 = predict_single_image(os.path.join(
        "../images",
        "00000761_004.png"))
    print(predict_1)
    predict_2 = predict_single_image(os.path.join("../images/00000002_000.png"))
    print(predict_2)
    predict_3 = predict_single_image(os.path.join("../images/00000003_000.png"))
    print(predict_3)
