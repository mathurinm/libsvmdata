# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause
import os
from pathlib import Path
from bz2 import BZ2Decompressor

import numpy as np
from scipy import sparse
from download import download
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file


NAMES = {
    'aloi': 'multiclass/aloi.bz2',
    'bibtex': 'multilabel/bibtex.bz2',
    'bodyfat': 'regression/bodyfat',
    'colon-cancer': 'binary/colon-cancer.bz2',
    'connect-4': 'multiclass/connect-4',
    'covtype.binary': 'binary/covtype.libsvm.binary.bz2',
    'delicious': 'multilabel/delicious.bz2',
    'dna': 'multiclass/dna.scale',
    'epsilon': 'binary/epsilon_normalized.bz2',
    'epsilon_test': 'binary/epsilon_normalized.t.bz2',
    'eunite2001': 'regression/eunite2001',
    'finance': 'regression/log1p.E2006.train.bz2',
    'finance-tf-idf': 'regression/E2006.train.bz2',
    'gisette': 'binary/gisette_scale.bz2',
    'glass': 'multiclass/glass.scale',
    'housing': 'regression/housing',
    'ijcnn1': 'binary/ijcnn1.bz2',
    'ijcnn1_test': 'binary/ijcnn1.t.bz2',
    'iris': 'multiclass/iris.scale',
    'kdda_train': 'binary/kdda.bz2',
    'letter': 'multiclass/letter.scale',
    'mediamill': 'multilabel/mediamill/train-exp1.svm.bz2',
    'mediamill_test': 'multilabel/mediamill/test-exp1.svm.bz2',
    'mnist': 'multiclass/mnist.bz2',
    'news20.binary': 'binary/news20.binary.bz2',
    'news20.multiclass': 'multiclass/news20.bz2',
    # 'protein': 'multiclass/protein.bz2',
    'rcv1.binary': 'binary/rcv1_train.binary.bz2',
    'rcv1.binary_test': 'binary/rcv1_test.binary.bz2',
    'rcv1.multiclass': 'multiclass/rcv1_train.multiclass.bz2',
    'rcv1.multiclass_test': 'multiclass/rcv1_test.multiclass.bz2',
    'rcv1_topics_test': 'multilabel/rcv1_topics_test_2.svm.bz2',
    'real-sim': 'binary/real-sim.bz2',
    'scene-classification': 'multilabel/scene_train.bz2',
    'scene-classification_test': 'multilabel/scene_test.bz2',
    'sector.scale': 'multiclass/sector/sector.scale.bz2',
    'sector.scale_test': 'multiclass/sector/sector.t.scale.bz2',
    'sector': 'multiclass/sector/sector.bz2',
    'sector_test': 'multiclass/sector/sector.t.bz2',
    'sensit': 'multiclass/vehicle/combined.bz2',
    'siam-competition2007': 'multilabel/tmc2007_train.svm.bz2',
    'siam-competition2007_test': 'multilabel/tmc2007_test.svm.bz2',
    'smallNORB': 'multiclass/smallNORB.bz2',
    'SUSY': 'binary/SUSY.bz2',
    'url': 'binary/url_combined.bz2',
    'usps': 'multiclass/usps.bz2',
    'usps_test': 'multiclass/usps.t.bz2',
    'webspam': 'binary/webspam_wc_normalized_trigram.svm.bz2',
    'yeast': 'multilabel/yeast_train.svm.bz2',
    'yeast_test': 'multilabel/yeast_test.svm.bz2',
}

N_FEATURES = {
    'aloi': 128,
    'bibtex': 1_836,
    'bodyfat': 14,
    'connect-4': 126,
    'covtype.binary': 54,
    'colon-cancer': 2_000,
    'delicious': 500,
    'dna': 180,
    'epsilon': 2_000,
    'epsilon_test': 2_000,
    'eunite2001': 16,
    'finance': 4_272_227,
    'finance-tf-idf': 150_360,
    'gisette': 5_000,
    'glass': 9,
    'housing': 13,
    'ijcnn1': 22,
    'ijcnn1_test': 22,
    'iris': 4,
    'kdda_train': 20_216_830,
    'letter': 16,
    'mediamill': 120,
    'mediamill_test': 120,
    'mnist': 780,
    'news20.binary': 1_355_191,
    'news20.multiclass': 62_061,
    # 'protein': 357,
    'rcv1.binary': 47_236,
    'rcv1.binary_test': 47_236,
    'rcv1.multiclass': 47_236,
    'rcv1.multiclass_test': 47_236,
    'rcv1_topics_test': 47_236,
    'real-sim': 20_958,
    'scene-classification': 294,
    'scene-classification_test': 294,
    'sector.scale': 55_197,
    'sector.scale_test': 55_197,
    'sector': 55_197,
    'sector_test': 55_197,
    'sensit': 100,
    'siam-competition2007': 30_438,
    'siam-competition2007_test': 30_438,
    'smallNORB': 18_432,
    'SUSY': 18,
    'url': 3_231_961,
    'usps': 7_291,
    'usps_test': 7_291,
    'webspam': 16_609_143,
    'yeast': 103,
    'yeast_test': 103,
}


# DATA_HOME is determined using environment variables.
# The top priority is the environment variable $LIBSVMDATA_HOME which is
# specific to this package.
# Else, it falls back on XDG_DATA_HOME if it is set.
# Finally, it defaults to $HOME/data.
# The data will be put in a subfolder 'libsvm'
def get_data_home():
    data_home = os.environ.get(
        'LIBSVMDATA_HOME', os.environ.get('XDG_DATA_HOME', None)
    )
    if data_home is None:
        data_home = Path.home() / 'data'

    return Path(data_home) / 'libsvm'


DATA_HOME = get_data_home()


def download_libsvm(dataset, destination, replace=False):
    """Download a dataset from LIBSVM website."""
    url = ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/" +
           NAMES[dataset])
    path = download(url, destination, replace=replace)
    return path


def _get_X_y(dataset, multilabel, replace=False):
    """Load a LIBSVM dataset as sparse X and observation y/Y.
    If X and y already exists as npz and npy, they are not redownloaded unless
    replace=True."""

    # some files are compressed, some are not:
    if NAMES[dataset].endswith('.bz2'):
        stripped_name = NAMES[dataset][:-4]
    else:
        stripped_name = NAMES[dataset]

    ext = '.npz' if multilabel else '.npy'
    y_path = DATA_HOME / f"{stripped_name}_target{ext}"
    X_path = DATA_HOME / f"{stripped_name}_data.npz"
    if replace or not y_path.exists() or not X_path.exists():
        tmp_path = DATA_HOME / stripped_name

        # Download the dataset
        source_path = DATA_HOME / NAMES[dataset]
        if not source_path.parent.exists():
            source_path.parent.mkdir(parents=True)
        download_libsvm(dataset, source_path, replace=replace)

        # decompress file only if it is compressed
        if NAMES[dataset].endswith('.bz2'):
            decompressor = BZ2Decompressor()
            print("Decompressing...")
            with open(tmp_path, "wb") as f, open(source_path, "rb") as g:
                for data in iter(lambda: g.read(100 * 1024), b''):
                    f.write(decompressor.decompress(data))
            source_path.unlink()

        n_features_total = N_FEATURES[dataset]

        print("Loading svmlight file...")
        with open(tmp_path, 'rb') as f:
            X, y = load_svmlight_file(
                f, n_features=n_features_total, multilabel=multilabel)

        tmp_path.unlink()
        X = sparse.csc_matrix(X)
        X.sort_indices()
        sparse.save_npz(X_path, X)

        if multilabel:
            indices = np.array([lab for labels in y for lab in labels])
            indptr = np.cumsum([0] + [len(labels) for labels in y])
            data = np.ones_like(indices)
            Y = sparse.csr_matrix((data, indices, indptr))
            sparse.save_npz(y_path, Y)
            return X, Y

        else:
            np.save(y_path, y)

    else:
        X = sparse.load_npz(X_path)
        if multilabel:
            y = sparse.load_npz(y_path)
        else:
            y = np.load(y_path)

    return X, y


def fetch_libsvm(dataset, replace=False, normalize=False, min_nnz=3):
    """
    Download a dataset from LIBSVM website.

    Parameters
    ----------
    dataset : string
        Dataset name. Must be in .NAMES.keys()

    replace : bool, default=False
        Whether to force download of dataset if already downloaded.

    normalize : bool, default=False
        If True, columns of X are set to unit norm. This may make little sense
        for a sparse matrix since centering is not performed.
        y is centered and set to unit norm if the dataset is a regression one.

    min_nnz: int, default=3
        Columns of X with strictly less than min_nnz non-zero entries are
        discarded.

    Returns
    -------
    X : scipy.sparse.csc_matrix
        Design matrix, in column sparse format.

    y : 1D or 2D np.array
        Design vector or matrix (in multiclass setting)


    References
    ----------
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

    """
    if dataset not in NAMES:
        raise ValueError("Unsupported dataset %s, check libsvmdata.supported "
                         "for the list of supported datasets" % dataset)
    multilabel = NAMES[dataset].split('/')[0] == 'multilabel'
    is_regression = NAMES[dataset].split('/')[0] == 'regression'

    print("Dataset: %s" % dataset)
    X, y = _get_X_y(dataset, multilabel, replace=replace)

    # preprocessing
    if min_nnz != 0:
        X = X[:, np.diff(X.indptr) >= min_nnz]

    if normalize:
        X = preprocessing.normalize(X, axis=0)
        if is_regression:
            y -= np.mean(y)
            y /= np.std(y)

    return X, y


if __name__ == "__main__":
    for dataset in NAMES:
        if not dataset.startswith("sector") and not dataset == "webspam":
            fetch_libsvm(dataset, replace=False)
