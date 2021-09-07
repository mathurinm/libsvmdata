import numpy as np

from libsvmdata import fetch_libsvm


# TODO: test all datasets without timeout
# @pytest.mark.parametrize("name", dataset.NAMES.keys())
# def test_datasets(name):
#     if "sector" in name:
#         pytest.xfail(name)

#     X, y = fetch_libsvm(name)
#     np.testing.assert_equal(X.shape[0], y.shape[0])


def test_binary():
    # download if not present:
    X, y = fetch_libsvm("news20.binary")
    np.testing.assert_equal(X.shape[0], y.shape[0])
    # also checks that loading saved files works:
    X, y = fetch_libsvm("news20.binary")


def test_multilabel():
    # test download
    X, Y = fetch_libsvm("rcv1_topics_test")
    np.testing.assert_equal(X.shape[0], Y.shape[0])
    # test saved npz loading
    X, Y = fetch_libsvm("rcv1_topics_test")


def test_regression():
    X, y = fetch_libsvm("bodyfat")
    np.testing.assert_equal(X.shape[0], y.shape[0])
    X, y = fetch_libsvm("bodyfat")


def test_multiclass():
    X, y = fetch_libsvm("iris")
    np.testing.assert_equal(X.shape[0], y.shape[0])
    X, y = fetch_libsvm("iris")


def test_wrong_dataset():
    np.testing.assert_raises(ValueError, fetch_libsvm, "unknowndataset")
