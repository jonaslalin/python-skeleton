import tensorflow as tf


def normalize(images):
    return images / 255


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (normalize(x_train), y_train), (normalize(x_test), y_test)
