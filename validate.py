import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

def validate_model(model, val_ds, class_names, threshold=0.1):
    true_labels = []
    predicted_labels = []

    for batch_idx, (images, labels) in enumerate(val_ds):
        preds = model.predict(images)
        # Convert labels from tensors to numpy and accumulate
        true_labels.extend(labels.numpy())  # Multi-hot labels
        predicted_labels.extend(preds)  # Raw sigmoid predictions

        if batch_idx >= 49:  # Limit for speed
            break

    # Convert true_labels and predicted_labels to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Convert predicted probabilities to binary labels
    predicted_labels = (predicted_labels > threshold).astype(int)

    # Compute classification report for each class independently
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names, zero_division=0))
def plot_training_history(history):
    # Loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history["binary_accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_binary_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.show()
