import os
from pathlib import Path
from .libsvm import DATASETS as libsvm_datasets
from .breheny import DATASETS as breheny_datasets

# General TODO :
# - when pyreadr is able to read R lists, use it instead of rpy2
# - check datasets such as mediamill that have several subfolders

# Comments for MM :
# - why removing columns with to few non zero entries when using sparse X ?
# - why XDG data home ?
# - transform data -> not our business ?
# - I remove the raw dataset file once data is extracted to .npy/.npz format
# - .bz2 decompression : why doing that ?

__version__ = "0.1dev"

ALL_DATABASES = libsvm_datasets + breheny_datasets
ALL_DATASETS = {dataset.dataset_name: dataset for dataset in ALL_DATABASES}


def print_supported_datasets():
    print("Supported datasets")
    print(
        "  LIBSVM :", ", ".join([dataset.dataset_name for dataset in libsvm_datasets])
    )
    print(
        "  Breheny :", ", ".join([dataset.dataset_name for dataset in breheny_datasets])
    )


def fetch_dataset(dataset_name, replace=False, verbose=False):
    """
    Download a dataset.

    Parameters
    ----------
    dataset_name : string
        Dataset name.

    replace : bool, default=False
        Whether to re-download the dataset if it is already downloaded.

    verbose : bool, default=False
        Whether or not to print information about dataset loading.


    Returns
    -------
    X : np.ndarray or scipy.sparse.csc_matrix
        Design matrix, as 2D array or column sparse format depending on the
        dataset.

    y : 1D or 2D np.ndarray
        Design vector (or matrix in multiclass setting).
    """

    if dataset_name not in ALL_DATASETS.keys():
        raise ValueError(
            f"Unsupported dataset {dataset_name}. Supported datasets can be "
            + "displayed using the print_supported_datasets() function."
        )

    dataset = ALL_DATASETS[dataset_name]

    X, y = dataset.get_X_y(replace=replace, verbose=verbose)

    return X, y
