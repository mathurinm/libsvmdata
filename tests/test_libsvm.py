import pytest
import numpy as np
from libsvmdata import fetch_dataset

# TODO : add more datasets to test ?
TEST_DATASETS = {
    "regression": ["abalone", "bodyfat"],
    "binary": ["a1a", "breast-cancer"],
    "multiclass": ["dna", "iris"],
    "multilabel": ["bibtex", "scene-classification"],
}


@pytest.mark.parametrize("dataset_name", TEST_DATASETS["regression"])
def test_regression(dataset_name):
    X, y = fetch_dataset(dataset_name)
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert X.shape[0] == y.shape[0]


@pytest.mark.parametrize("dataset_name", TEST_DATASETS["binary"])
def test_binary(dataset_name):
    X, y = fetch_dataset(dataset_name)
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert X.shape[0] == y.shape[0]
    assert len(np.unique(y)) == 2.0


@pytest.mark.parametrize("dataset_name", TEST_DATASETS["multiclass"])
def test_multiclass(dataset_name):
    X, y = fetch_dataset(dataset_name)
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert X.shape[0] == y.shape[0]
    assert len(np.unique(y)) > 2.0


@pytest.mark.parametrize("dataset_name", TEST_DATASETS["multilabel"])
def test_multilabel(dataset_name):
    X, y = fetch_dataset(dataset_name)
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert X.shape[0] == y.shape[0]
    assert y.shape[1] > 2
