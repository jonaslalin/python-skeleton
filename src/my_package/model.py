import tensorflow as tf


def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=10, activation="relu"),
            tf.keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )
    return model
