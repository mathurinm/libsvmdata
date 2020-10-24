import numpy as np
import pytest

from libsvmdata import fetch_libsvm
from libsvmdata.datasets import NAMES


@pytest.mark.parametrize("name", NAMES.keys())
def test_datasets(name):
    if "sector" in name:
        pytest.xfail(name)

    X, y = fetch_libsvm(name)
    np.testing.assert_equal(X.shape[0], y.shape[0])
