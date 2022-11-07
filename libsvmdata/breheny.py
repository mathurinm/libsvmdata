import numpy as np
import rpy2.robjects as robjects
from scipy import sparse
from .core import _get_data_home, AbstractDataset

BREHENY_DATA_HOME = _get_data_home("breheny")
BREHENY_BASE_URL = "https://s3.amazonaws.com/pbreheny-data-sets"


class BrehenyDataset(AbstractDataset):
    def __init__(
        self,
        dataset_name,
        dataset_file,
        X_key,
        y_key,
        transpose_X=False,
        transpose_y=False,
    ):

        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.dataset_dir = BREHENY_DATA_HOME
        self.dataset_url = "/".join([BREHENY_BASE_URL, dataset_file])
        self.X_key = X_key
        self.y_key = y_key
        self.transpose_X = transpose_X
        self.transpose_y = transpose_y

    def _load_file_and_save_data(self, raw_dataset_path, ext_dataset_path):

        readRDS = robjects.r["readRDS"]
        rdsfile = readRDS(str(raw_dataset_path))
        rdsfile_dict = dict(zip(rdsfile.names, list(rdsfile)))
        X = np.asarray(rdsfile_dict[self.X_key])
        y = np.asarray(rdsfile_dict[self.y_key])
        raw_dataset_path.unlink()

        if self.transpose_X:
            X = X.T

        if self.transpose_y:
            y = y.T

        X_path = str(ext_dataset_path) + "_X"
        if len(X.data) >= 0.5 * X.shape[0] * X.shape[1]:
            X = X.toarray(order="F")
            np.save(X_path, X)
        else:
            X = sparse.csc_matrix(X)
            X.sort_indices()
            sparse.save_npz(X_path, X)

        y_path = str(ext_dataset_path) + "_y"
        if sparse.issparse(y):
            sparse.save_npz(y_path, y)
        else:
            np.save(y_path, y)

        return X, y


DATASETS = [
    BrehenyDataset("Golub1999", "Golub1999.rds", "X", "y"),
    BrehenyDataset("Singh2002", "Singh2002.rds", "X", "y"),
    BrehenyDataset(
        "Gode2011", "Gode2011.rds", "Y", "expCond", transpose_X=True, transpose_y=True
    ),
    BrehenyDataset(
        "Scholtens2004",
        "Scholtens2004.rds",
        "Y",
        "expCond",
        transpose_X=True,
        transpose_y=True,
    ),
    BrehenyDataset("pollution", "pollution.rds", "X", "y"),
    BrehenyDataset("whoari", "whoari.rds", "X", "y"),
    BrehenyDataset("bcTCGA", "bcTCGA.rds", "X", "y"),
    BrehenyDataset("Koussounadis2014", "Koussounadis2014.rds", "X", "y"),
    BrehenyDataset("Scheetz2006", "Scheetz2006.rds", "X", "y"),
    BrehenyDataset("Ramaswamy2001", "Ramaswamy2001.rds", "X", "y"),
    BrehenyDataset("Ramaswamy2001_test", "Ramaswamy2001.rds", "X.test", "y.test"),
    BrehenyDataset("Shedden2008_survival", "Shedden2008.rds", "X", "S"),
    BrehenyDataset(
        "Shedden2008_covariates", "Shedden2008.rds", "X", "Z", transpose_y=True
    ),
    BrehenyDataset("Rhee2006", "Rhee2006.rds", "X", "y"),
    BrehenyDataset("Yeoh2002", "Yeoh2002.rds", "X", "y"),
    BrehenyDataset("glc-amd", "glc-amd.rds", "X", "y"),
    BrehenyDataset("spam", "spam.rds", "X", "y"),
    BrehenyDataset("spam_test", "spam.rds", "Xtest", "ytest"),
]
