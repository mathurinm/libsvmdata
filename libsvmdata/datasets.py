# This file aims to avoid compatibility issues with versions of the libsvmdata
# package anterior to https://github.com/mathurinm/libsvmdata/pull/37.

import download
import numpy as np
import warnings
from sklearn import preprocessing
from scipy import sparse
from .libsvm import DATASETS as libsvm_datasets
from .core import fetch_dataset

# The `NAMES` variable before the pull request #37 can be reconstructed from
# the `DATASETS` variable in the `libsvm.py` file.
NAMES = {
    dataset.dataset_name: "/".join([dataset.task_name, dataset.dataset_file])
    for dataset in libsvm_datasets
}


def download_libsvm(dataset, destination, replace=False, verbose=False):
    """Download a dataset from LIBSVM website."""

    warnings.warn(
        "The function `download_libsvm` will be depreciated in `v0.5` and "
        "removed in `v0.6`. See "
        "https://github.com/mathurinm/libsvmdata/pull/37 for more details.",
        FutureWarning
    )

    url = (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
        + NAMES[dataset]
    )
    path = download(url, destination, replace=replace, verbose=verbose)
    return path


def fetch_libsvm(dataset, replace=False, normalize=False, min_nnz=0,
                 verbose=False):
    """
    Download a dataset from LIBSVM website.
    Parameters
    ----------
    dataset : string
        Dataset name. Must be in libsvmdata.supported.
    replace : bool, default=False
        Whether to force download of dataset if already downloaded.
    normalize : bool, default=False
        If True, columns of X are set to unit norm. This may make little sense
        for a sparse matrix since centering is not performed.
        y is centered and set to unit norm if the dataset is a regression one.
    min_nnz : int, default=0
        When X is sparse, columns of X with strictly less than min_nnz
        non-zero entries are discarded.
    verbose : bool, default=False
        Whether or not to print information about dataset loading.
    Returns
    -------
    X : np.ndarray or scipy.sparse.csc_matrix
        Design matrix, as 2D array or column sparse format depending on the
        dataset.
    y : 1D or 2D np.ndarray
        Design vector or matrix (in multiclass setting).
    References
    ----------
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
    """

    warnings.warn(
        "The function `fetch_libsvm` will be depreciated in `v0.5` and "
        "replaced by `fetch_dataset`. It will be removed in `v0.6`. See "
        "https://github.com/mathurinm/libsvmdata/pull/37 for more details.",
        FutureWarning
    )

    if dataset not in NAMES:
        raise ValueError("Unsupported dataset %s. " % dataset +
                         "Supported datasets are: \n" + ', '.join(NAMES))
    is_regression = NAMES[dataset].split('/')[0] == 'regression'

    if verbose:
        print("Dataset: %s" % dataset)

    # Does exactly the same as the original `_get_X_y` function but without
    # the normalization and the removing of too sparse columns done in
    # post-processing step when. These steps are therefore done just below.
    X, y = fetch_dataset(dataset, replace=replace, verbose=verbose)

    # removing columns with to few non zero entries when using sparse X
    if sparse.issparse(X) and min_nnz != 0:
        X = X[:, np.diff(X.indptr) >= min_nnz]

    if normalize:
        X = preprocessing.normalize(X, axis=0)
        if is_regression:
            y -= np.mean(y)
            y /= np.std(y)

    return X, y
