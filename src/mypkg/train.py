import tensorflow as tf

from mypkg.data import load_data
from mypkg.model import create_model


def train_model(epochs=5):
    model = create_model()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    (x_train, y_train), (x_test, y_test) = load_data()
    model.fit(x_train, y_train, epochs=epochs)
    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    train_model()
