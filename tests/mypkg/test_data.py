from mypkg.data import load_data


def test_load_data():
    (x_train, y_train), (x_test, y_test) = load_data()
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)
    assert x_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)
