import numpy as np

from mypkg.model import create_model


def load_data():
    rng = np.random.default_rng()
    return rng.uniform(size=(32, 28, 28))


def test_create_model():
    model = create_model()
    assert model(load_data()).numpy().shape == (32, 10)
