import matplotlib.pyplot as plt
import tensorflow as tf

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(5):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255)

        label = label_batch[n].numpy() if isinstance(label_batch[n], tf.Tensor) else label_batch[n]
        plt.title(label)
        plt.axis("off")

    plt.show()