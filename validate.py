import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

def validate_model(model, val_ds, class_names, threshold=0.5):
    true_labels = []
    predicted_labels = []

    for batch_idx, (images, labels) in enumerate(val_ds):
        preds = model.predict(images)
        print(f"First few predictions: {preds[:5]}")
        # Convert labels from tensors to numpy
        true_labels.extend(labels.numpy().tolist())  # Multi-hot labels
        predicted_labels.extend(preds.tolist())  # Raw sigmoid predictions

        if batch_idx >= 49:  # Limit for speed
            break
    # Convert predicted probabilities to binary labels
    predicted_labels = [[1 if prob > threshold else 0 for prob in row] for row in predicted_labels]

    # **Convert Multi-Hot to Class Names**
    def multi_hot_to_labels(multi_hot):
        return [class_names[i] for i in range(len(class_names)) if i < len(multi_hot) and multi_hot[i] == 1]

    # Ensure labels exist
    true_labels_list = [multi_hot_to_labels(row) for row in true_labels if sum(row) > 0]
    predicted_labels_list = [multi_hot_to_labels(row) for row in predicted_labels if sum(row) > 0]

    # Debugging: Print sample converted labels
    print(f"First 5 true labels (converted): {true_labels_list[:5]}")
    print(f"First 5 predicted labels (converted): {predicted_labels_list[:5]}")

    # Flatten labels
    true_flat = [label for sublist in true_labels_list for label in sublist]
    pred_flat = [label for sublist in predicted_labels_list for label in sublist]

    # Print classification report
    print("Classification Report:")
    print(classification_report(true_flat, pred_flat, target_names=class_names, zero_division=0))


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
