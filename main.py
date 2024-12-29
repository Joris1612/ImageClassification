from train import train_model, val_ds, unique_labels
from validate import validate_model, plot_training_history
import tensorflow as tf

def main():
    #print("Starting training...")
    #history, trained_model = train_model()
    #print("Training completed.")
    # trained_model.save("XRAY-model-E100.keras")

    # Load the saved model
    print("Loading the trained model...")
    trained_model = tf.keras.models.load_model("XRAY-model-E100.keras")

    print("Starting validation...")
    validate_model(trained_model, val_ds, unique_labels)

if __name__ == "__main__":
    main()
