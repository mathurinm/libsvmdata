import pytest
import numpy as np
from libsvmdata import fetch_dataset
from libsvmdata.breheny import DATASETS as BREHENY_DATASETS

TEST_DATASETS = [dataset.dataset_name for dataset in BREHENY_DATASETS]


@pytest.mark.parametrize("dataset_name", TEST_DATASETS)
def test_dataset(dataset_name):
    X, y = fetch_dataset(dataset_name)
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert X.shape[0] == y.shape[0]
