import pytest
import numpy as np
from libsvmdata import fetch_dataset, print_supported_datasets


# TODO : add other datasets to test ?
def test_replace():
    X_first, y_first = fetch_dataset("iris", replace=True)
    X_second, y_second = fetch_dataset("iris")
    np.testing.assert_equal(X_first, X_second)
    np.testing.assert_equal(y_first, y_second)


def test_wrong_dataset():
    with pytest.raises(ValueError):
        fetch_dataset("unknowndataset")


def test_print_supported_datasets():
    print_supported_datasets()
