import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from libsvmdata import fetch_libsvm

TEST_DATASETS = {
    "regression": [
        ("abalone", 4_177, 8),
        ("bodyfat", 252, 14),
    ],
    "binary": [
        ("a1a", 1_605, 123),
        ("breast-cancer", 683, 10),
    ],
    "multiclass": [
        ("dna", 2_000, 180),
        ("iris", 150, 4),
    ],
    "multilabel": [
        ("bibtex", 7_395, 1_836),
        ("scene-classification", 1_211, 294),
    ],
}


@pytest.mark.filterwarnings("ignore:FutureWarning")
@pytest.mark.parametrize("dataset_name,n,p", TEST_DATASETS["regression"])
def test_regression(dataset_name, n, p):
    X, y = fetch_libsvm(dataset_name)
    assert_equal(X.shape[0], n)
    assert_equal(X.shape[1], p)
    assert_equal(y.shape[0], n)


@pytest.mark.filterwarnings("ignore:FutureWarning")
@pytest.mark.parametrize("dataset_name,n,p", TEST_DATASETS["binary"])
def test_binary(dataset_name, n, p):
    X, y = fetch_libsvm(dataset_name, n, p)
    assert_equal(X.shape[0], n)
    assert_equal(X.shape[1], p)
    assert_equal(y.shape[0], n)
    assert_equal(len(np.unique(y)), 2)


@pytest.mark.filterwarnings("ignore:FutureWarning")
@pytest.mark.parametrize("dataset_name,n,p", TEST_DATASETS["multiclass"])
def test_multiclass(dataset_name, n, p):
    X, y = fetch_libsvm(dataset_name)
    assert_equal(X.shape[0], n)
    assert_equal(X.shape[1], p)
    assert_equal(y.shape[0], n)
    assert_(len(np.unique(y)) > 2)


@pytest.mark.filterwarnings("ignore:FutureWarning")
@pytest.mark.parametrize("dataset_name,n,p", TEST_DATASETS["multilabel"])
def test_multilabel(dataset_name, n, p):
    X, y = fetch_libsvm(dataset_name)
    assert_equal(X.shape[0], n)
    assert_equal(X.shape[1], p)
    assert_equal(y.shape[0], n)
    assert_(y.shape[1] > 2)


@pytest.mark.filterwarnings("ignore:FutureWarning")
def test_normalization():
    X, y = fetch_libsvm("abalone", normalize=True)
    assert_allclose(np.linalg.norm(X, axis=0), 1.)
    assert_allclose(np.mean(y), 0., atol=1e-07)
    assert_allclose(np.std(y), 1.)
