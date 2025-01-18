import tensorflow as tf
import pandas as pd
import os
import keras
from keras import layers, mixed_precision
import numpy as np

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMAGE_SIZE = [128, 128]

# Define class names manually
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Map class names to indices
label_to_int_mapping = {name: idx for idx, name in enumerate(class_names)}

# Load the CSV file
label_df = pd.read_csv(r"C:\Users\mathi\OneDrive\Documents\School\Dataset\Data_Entry_2017_v2020.csv")

# Strip extra spaces from column names
label_df.columns = label_df.columns.str.strip()

# Dictionary to store multi-label classification
label_dict = {}

for idx, row in label_df.iterrows():
    image_name = os.path.basename(row["Image Index"]).strip().lower()
    labels = row["Finding Labels"].split("|")

    # Multi-hot encoding
    multi_hot_vector = np.zeros(len(class_names), dtype=np.int32)
    for label in labels:
        label = label.strip()
        if label in class_names:
            multi_hot_vector[label_to_int_mapping[label]] = 1

    label_dict[image_name] = multi_hot_vector


def get_label(file_path):
    """Extract filename and return multi-hot encoded label."""
    image_name = os.path.basename(file_path.numpy().decode("utf-8")).strip().lower()
    return label_dict.get(image_name, np.zeros(len(class_names), dtype=np.int32))


def tf_get_label(file_path):
    """Use tf.py_function to wrap the Python function `get_label`."""
    label = tf.py_function(func=get_label, inp=[file_path], Tout=tf.int32)
    label.set_shape([len(class_names)])  # Ensure shape consistency
    return label


def decode_img(img):
    """Decode and resize image."""
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    return img


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])

    label = tf_get_label(file_path)

    return img, label


# Set batch size
AUTOTUNE = tf.data.AUTOTUNE

# Create TensorFlow dataset
ds = tf.data.Dataset.list_files(r"C:\Users\mathi\OneDrive\Documents\School\Dataset\images\*.png")

# Convert the dataset to a list to check its size (no mapping or batching yet)
ds_list = list(ds)

# Define the split sizes
total_images = len(ds_list)
train_size = int(0.8 * total_images)
val_size = total_images - train_size

# Split dataset into training and validation sets **before** applying map and batch
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size).take(val_size)

# Apply map before batching
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# Apply batching and prefetching after mapping
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def conv_block(filters, inputs):
    x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(inputs)
    x = layers.SeparableConv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    # Use a smaller pool size to prevent negative dimensions
    outputs = layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding="valid")(x)
    return outputs


def dense_block(units, dropout_rate, inputs):
    x = layers.Dense(units, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dropout(dropout_rate)(x)

    return outputs

''' Homemade model
def build_model():
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = conv_block(8, x)
    x = layers.GlobalAveragePooling2D()(x)
    x = dense_block(32, 0.5, x)
    outputs = layers.Dense(len(class_names), activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
'''


def build_model():
    base_model = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    # Freeze the base model
    base_model.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(len(class_names), activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    return model


def get_checkpoint_callback(name):
    checkpoint_cb = keras.callbacks.ModelCheckpoint(name, save_best_only=True, verbose=1)
    return checkpoint_cb


early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
# Example of using ExponentialDecay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


def main_train(epochs, name):
    checkpoint_cb = get_checkpoint_callback(name)
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    with strategy.scope():
        model = build_model()
        model.compile(
            optimizer='adam',
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )
    model.save(name)
    return history, model


# Prevent automatic execution when imported
if __name__ == "__main__":
    history, model = main_train(epochs=1, name="test.keras")


def train_model(epochs, name):
    return main_train(epochs, name)
