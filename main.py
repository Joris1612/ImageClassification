from train import train_model, val_ds, class_names
from validate import validate_model, plot_training_history
import tensorflow as tf

def main():
    epochs = 5
    name = "XRAY-E5.keras"
    print("Starting training...")
    train_model(epochs, name)
    print("Training completed.")
    # Load the saved model
    print("Loading the trained model...")
    trained_model = tf.keras.models.load_model(name)

    print("Starting validation...")
    validate_model(trained_model, val_ds, class_names)

if __name__ == "__main__":
    main()
