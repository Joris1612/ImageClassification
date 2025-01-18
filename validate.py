import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

#main validation method
def validate_model(model, val_ds, class_names, threshold=0.1):
    true_labels = []
    predicted_labels = []

    for batch_idx, (images, labels) in enumerate(val_ds):
        preds = model.predict(images)
        # Convert labels from tensors to numpy and accumulate
        true_labels.extend(labels.numpy())
        predicted_labels.extend(preds)

        if batch_idx >= 49:
            break

    # Convert true_labels and predicted_labels to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Convert predicted probabilities to binary labels
    predicted_labels = (predicted_labels > threshold).astype(int)

    # Compute classification report for each class independently
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names, zero_division=0))
