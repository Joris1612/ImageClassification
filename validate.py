from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def validate_model(model, val_ds, class_names):

    # Ensure valid labels are indexed correctly
    valid_label_indices = {idx: label for idx, label in enumerate(class_names)}

    # Initialize lists for true and predicted labels
    true_labels = []
    predicted_labels = []

    for batch_idx, (images, labels) in enumerate(val_ds):
        # Predict on batch
        preds = model.predict(images)

        # Process true and predicted labels
        true_labels.extend(tf.argmax(labels, axis=1).numpy())
        predicted_labels.extend(tf.argmax(preds, axis=1).numpy())

        # Stop after 50 batches (optional, to limit validation time)
        if batch_idx >= 49:
            break

    # Debugging: Print label counts
    print(f"Total true labels collected: {len(true_labels)}")
    print(f"Total predicted labels collected: {len(predicted_labels)}")

    # Convert integer indices to string labels using `class_names`
    true_labels_str = [valid_label_indices.get(label, "unknown") for label in true_labels]
    predicted_labels_str = [valid_label_indices.get(pred, "unknown") for pred in predicted_labels]

    # Filter out unknown labels
    filtered_true_labels = [
        label for label in true_labels_str if label != "unknown"
    ]
    filtered_predicted_labels = [
        pred for pred, true in zip(predicted_labels_str, true_labels_str) if true != "unknown"
    ]

    # Check if filtered labels are empty
    if not filtered_true_labels or not filtered_predicted_labels:
        print("No valid labels found for classification report.")
        return

    # Generate classification report
    print("Classification Report:")
    print(classification_report(
        filtered_true_labels,
        filtered_predicted_labels,
        labels=class_names,
        target_names=class_names
    ))

    from collections import Counter
    print("True label distribution:", Counter(true_labels_str))
    print("Predicted label distribution:", Counter(predicted_labels_str))

    for images, labels in val_ds.take(1):
        preds = model.predict(images)
        print("Raw predictions:", preds[:5])
        print("True labels (one-hot):", labels[:5])

    # Print final classification report (with zero_division=0 to prevent warnings for zero samples)
    print(classification_report(
        filtered_true_labels,
        filtered_predicted_labels,
        labels=class_names,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(filtered_true_labels, filtered_predicted_labels, labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    # Annotate confusion matrix
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value}', ha='center', va='center', color='white', fontsize=12)

    # Add axis labels and title
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.show()

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
