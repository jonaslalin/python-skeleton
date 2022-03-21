import tensorflow as tf

from mypkg.data import load_data
from mypkg.model import create_model


def train_model(batch_size=32, epochs=5):
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size, epochs)
    model.evaluate(x_test, y_test, batch_size)


if __name__ == "__main__":
    for batch_size in [2**exponent for exponent in range(5, 15)]:
        print(f"Training with batch size {batch_size}")
        train_model(batch_size)
