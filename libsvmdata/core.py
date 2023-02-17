from libsvmdata.libsvm import DATASETS as libsvm_datasets

ALL_DATABASES = {"LIBSVM": libsvm_datasets}

ALL_DATASETS = {
    dataset.dataset_name: dataset
    for datasets in ALL_DATABASES.values()
    for dataset in datasets
}


def fetch_dataset(dataset_name, replace=False, verbose=False):
    """
    Load a dataset. It is downloaded only if not present or when replace=True.

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
            f"Unsupported dataset `{dataset_name}`. Supported datasets can be "
            "displayed using the `libsvmdata.print_supported_datasets` "
            "function."
        )

    dataset = ALL_DATASETS[dataset_name]

    X, y = dataset.get_X_y(replace=replace, verbose=verbose)

    return X, y


def print_supported_datasets():
    print("Supported datasets")
    for database_name, datasets in ALL_DATABASES.items():
        print(f"- {database_name}: ")
        print(", ".join(dataset.dataset_name for dataset in datasets))
