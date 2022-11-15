import os
import re
import numpy as np
from abc import ABC, abstractmethod
from download import download
from pathlib import Path
from scipy import sparse


def _get_data_home(subdir_name=""):
    """
    Defines the data home folder. The top priority is the environment
    variable $LIBSVMDATA_HOME which is specific to this package. Otherwise, we
    seek for the variable $XDG_DATA_HOME. Finally, the fallback is $HOME/data.
    """
    data_home = os.environ.get("LIBSVMDATA_HOME", None)
    if data_home is None:
        data_home = os.environ.get("XDG_DATA_HOME", None)
    if data_home is None:
        data_home = Path.home() / "data"
    return data_home / subdir_name


class AbstractDataset(ABC):
    """Base class defining a dataset along with its fetching methods."""

    # In the derived class, __init__() must set the following attributes :
    dataset_name = None  # dataset name
    dataset_file = None  # dataset file (with potential extensions)
    dataset_dir = None  # subdirectory name (see _get_data_home())
    dataset_url = None  # dataset download url

    @abstractmethod
    def __init__(self):
        """
        In the derived class, this function must define the class attributes.
        It can also be used to pass additional information required in the
        function _load_file_and_save_data() of the derived class.
        """
        pass

    @abstractmethod
    def _load_file_and_save_data(self, raw_dataset_path, ext_dataset_path):
        """
        In the derived class, this function is responsible of the
        transformation of the raw dataset file into two .npy/.npz files
        containing the feature matrix X and the response vector/matrix y. These
        files must be named <self.dataset_name>_X.<npz/npy> and
        <self.dataset_name>_y.<npz/npy>. This function is also responsible for
        removing the raw dataset file when needed.
        """
        pass

    def _load_data(self, ext_dataset_path):
        """Load data from the extracted .npz/.npy files."""

        try:
            X = sparse.load_npz(str(ext_dataset_path) + "_X.npz")
        except FileNotFoundError:
            X = np.load(str(ext_dataset_path) + "_X.npy")

        try:
            y = sparse.load_npz(str(ext_dataset_path) + "_y.npz")
        except FileNotFoundError:
            y = np.load(str(ext_dataset_path) + "_y.npy")

        return X, y

    def get_X_y(self, replace=False, verbose=False):
        """
        Load a dataset as matrix X and vector y. If X and y already exist as
        .npz and/or .npy files, they are not redownloaded, unless replace=True.
        """

        raw_dataset_path = self.dataset_dir / self.dataset_file
        ext_dataset_path = self.dataset_dir / self.dataset_name

        # Check if the dataset already exists
        if self.dataset_dir.exists():
            regex = re.compile(f"{self.dataset_name}_(X|y).(npz|npy)")
            files = os.listdir(self.dataset_dir)
            found = [f for f in files if re.search(regex, f)]
            exists = len(found) == 2
        else:
            found = []
            exists = False

        if replace or not exists:

            # Remove existing dataset files if there are any
            if raw_dataset_path.exists():
                raw_dataset_path.unlink()
            for file in found:
                Path(self.dataset_dir / file).unlink()

            # Path of the raw dataset file
            if verbose:
                print("Downloading...")
            download(
                self.dataset_url,
                raw_dataset_path,
                progressbar=verbose,
                replace=replace,
                verbose=verbose,
            )

            if verbose:
                print("Loading file and saving data...")
            X, y = self._load_file_and_save_data(
                raw_dataset_path, ext_dataset_path
            )

        else:
            if verbose:
                print("Loading data...")
            X, y = self._load_data(ext_dataset_path)

        return X, y
