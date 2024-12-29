import tensorflow as tf
import pandas as pd
import os
import keras
from keras import layers

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 4 * strategy.num_replicas_in_sync
IMAGE_SIZE = [128, 128]

# Load the CSV file
label_df = pd.read_csv(r"C:\Users\mathi\OneDrive\Documents\School\Dataset\Data_Entry_2017_v2020.csv")

# Strip extra spaces from column names
label_df.columns = label_df.columns.str.strip()

# Normalize keys and values in label_dict
label_dict = {
    os.path.basename(name).strip().lower(): label.strip()
    for name, label in zip(label_df["Image Index"], label_df["Finding Labels"])
}

# Get all unique labels from the label_dict and create an integer mapping
unique_labels = sorted(set(label_dict.values()))
label_to_int_mapping = {label: idx for idx, label in enumerate(unique_labels)}


# Print out the mapping for debugging
# print(label_to_int_mapping)

def get_label(file_path):
    # Extract the filename from the full path
    image_name = os.path.basename(file_path.numpy().decode("utf-8")).strip().lower()

    # Use the label_dict to get the string label, then convert it to an integer
    label_string = label_dict.get(image_name, "unknown")
    return label_to_int_mapping.get(label_string, -1)


def tf_get_label(file_path):
    return tf.py_function(func=get_label, inp=[file_path], Tout=tf.int32)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img


def process_path(file_path):
    label = tf_get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, tf.reshape(label, [])  # Ensure label is a scalar tensor

@tf.function
def preprocess_dataset(image, label):
    label = tf.one_hot(label, depth=len(unique_labels))
    return image, label


# Create the TensorFlow dataset
ds = tf.data.Dataset.list_files(r"C:\Users\mathi\OneDrive\Documents\School\Dataset\images\*.png")
ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)

# Apply the preprocess function
ds = ds.map(preprocess_dataset)

ds = ds.shuffle(10000)
train_ds = ds.take(4200)
val_ds = ds.skip(4200)


def prepare_for_training(ds, cache=False):
    # If not caching, skip the caching step
    if cache:
        # Only cache if needed (for small datasets or if you are sure you have enough memory)
        ds = ds.cache()

    # Perform batching
    ds = ds.batch(BATCH_SIZE)

    # Prefetch for asynchronous data loading
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


ds = ds.repeat()
train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)
image_batch, label_batch = next(iter(train_ds))
os.environ['KERAS_BACKEND'] = 'tensorflow'


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


def build_model():
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = conv_block(16, x)
    x = conv_block(32, x)
    x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = dense_block(128, 0.5, x)
    x = dense_block(64, 0.3, x)
    outputs = layers.Dense(len(unique_labels), activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

checkpoint_cb = keras.callbacks.ModelCheckpoint("XRAY-model-E100.keras", save_best_only=True)

early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
# Example of using ExponentialDecay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

#update training method calling

def main_train():
    with strategy.scope():
        model = build_model()
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )
    history = model.fit(
        train_ds,
        epochs=100,
        validation_data=val_ds,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )
    return history, model

# Prevent automatic execution when imported
if __name__ == "__main__":
    history, model = main_train()


def train_model():
    return main_train()
