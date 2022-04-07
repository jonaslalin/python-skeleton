import pytest
import tensorflow as tf

from my_package.model import create_model

g = tf.random.Generator.from_seed(1)


@pytest.fixture
def inputs():
    return g.uniform(shape=(32, 5))


def test_create_model(inputs):
    model = create_model()
    outputs = model(inputs)
    assert outputs.shape == (32, 1)
