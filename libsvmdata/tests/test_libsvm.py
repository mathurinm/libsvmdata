import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from libsvmdata import fetch_dataset

# TODO : add more datasets to test ?
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


@pytest.mark.parametrize("dataset_name,n,p", TEST_DATASETS["regression"])
def test_regression(dataset_name, n, p):
    X, y = fetch_dataset(dataset_name)
    assert_equal(X.shape[0], n)
    assert_equal(X.shape[1], p)
    assert_equal(y.shape[0], n)


@pytest.mark.parametrize("dataset_name,n,p", TEST_DATASETS["binary"])
def test_binary(dataset_name, n, p):
    X, y = fetch_dataset(dataset_name, n, p)
    assert_equal(X.shape[0], n)
    assert_equal(X.shape[1], p)
    assert_equal(y.shape[0], n)
    assert_equal(len(np.unique(y)), 2)


@pytest.mark.parametrize("dataset_name,n,p", TEST_DATASETS["multiclass"])
def test_multiclass(dataset_name, n, p):
    X, y = fetch_dataset(dataset_name)
    assert_equal(X.shape[0], n)
    assert_equal(X.shape[1], p)
    assert_equal(y.shape[0], n)
    assert_(len(np.unique(y)) > 2)


@pytest.mark.parametrize("dataset_name,n,p", TEST_DATASETS["multilabel"])
def test_multilabel(dataset_name, n, p):
    X, y = fetch_dataset(dataset_name)
    assert_equal(X.shape[0], n)
    assert_equal(X.shape[1], p)
    assert_equal(y.shape[0], n)
    assert_(y.shape[1] > 2)
