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
    'a1a': 'binary/a1a',
    'a1a_test': 'binary/a1a.t',
    'a2a': 'binary/a2a',
    'a2a_test': 'binary/a2a.t',
    'a3a': 'binary/a1a',
    'a3a_test': 'binary/a1a.t',
    'a4a': 'binary/a1a',
    'a4a_test': 'binary/a1a.t',
    'a5a': 'binary/a1a',
    'a5a_test': 'binary/a1a.t',
    'a6a': 'binary/a1a',
    'a6a_test': 'binary/a1a.t',
    'a7a': 'binary/a7a',
    'a7a_test': 'binary/a7a.t',
    'a8a': 'binary/a8a',
    'a8a_test': 'binary/a8a.t',
    'a9a': 'binary/a9a',
    'a9a_test': 'binary/a9a.t',
    'abalone': 'regression/abalone',
    'abalone_scale': 'regression/abalone_scale',
    'aloi': 'multiclass/aloi.bz2',
    'australian': 'binary/australian',
    'australian_scale': 'binary/australian_scale',
    'bibtex': 'multilabel/bibtex.bz2',
    'bodyfat': 'regression/bodyfat',
    'breast-cancer': 'binary/breast-cancer',
    'breast-cancer_scale': 'binary/breast-cancer_scale',
    'cadata': 'regression/cadata',
    'cifar10': 'multiclass/cifar10.bz2',
    'cifar10_test': 'multiclass/cifar10.t.bz2',
    'cod-rna': 'binary/cod-rna',
    'cod-rna_test': 'binary/cod-rna.t',
    'colon-cancer': 'binary/colon-cancer.bz2',
    'connect-4': 'multiclass/connect-4',
    'covtype.binary': 'binary/covtype.libsvm.binary.bz2',
    'covtype.multiclass': 'multiclass/covtype.bz2',
    'covtype.multiclass_scale': 'multiclass/covtype.scale01.bz2',
    'cpusmall': 'regression/cpusmall',
    'delicious': 'multilabel/delicious.bz2',
    'diabetes': 'binary/diabetes',
    'diabetes_scale': 'binary/diabetes_scale',
    'dna': 'multiclass/dna.scale',
    'duke breast-cancer': 'binary/duke.bz2',
    'epsilon': 'binary/epsilon_normalized.bz2',
    'epsilon_test': 'binary/epsilon_normalized.t.bz2',
    'eunite2001': 'regression/eunite2001',
    'finance': 'regression/log1p.E2006.train.bz2',
    'finance-tf-idf': 'regression/E2006.train.bz2',
    'fourclass': 'binary/fourclass',
    'fourclass_scale': 'binary/fourclass_scale',
    'german.numer': 'binary/german.numer',
    'german.numer_scale': 'binary/german.numer_scale',
    'gisette': 'binary/gisette_scale.bz2',
    'glass': 'multiclass/glass.scale',
    'heart': 'binary/heart',
    'heart_scale': 'binary/heart_scale',
    'HIGGS': 'binary/HIGGS.bz2',
    'housing': 'regression/housing',
    'ijcnn1': 'binary/ijcnn1.bz2',
    'ijcnn1_test': 'binary/ijcnn1.t.bz2',
    'ionosphere': 'binary/ionosphere_scale',
    'iris': 'multiclass/iris.scale',
    'kdda_train': 'binary/kdda.bz2',
    'letter': 'multiclass/letter.scale',
    'leukemia': 'binary/leu.bz2',
    'leukemia_test': 'binary/leu.t.bz2',
    'liver-disorders': 'binary/liver-disorders',
    'liver-disorders_scale': 'binary/liver-disorders_scale',
    'liver-disorders_test': 'binary/liver-disorders.t',
    'madelon': 'binary/madelon',
    'madelon_test': 'binary/madelon.t',
    'mediamill': 'multilabel/mediamill/train-exp1.svm.bz2',
    'mediamill_test': 'multilabel/mediamill/test-exp1.svm.bz2',
    'mnist': 'multiclass/mnist.bz2',
    'news20.binary': 'binary/news20.binary.bz2',
    'news20.multiclass': 'multiclass/news20.bz2',
    'pendigits': 'multiclass/pendigits',
    'pendigits_test': 'multiclass/pendigits.t',
    'phishing': 'binary/phishing',
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
    'skin_nonskin': 'binary/skin_nonskin',
    'smallNORB': 'multiclass/smallNORB.bz2',
    'sonar': 'binary/sonar_scale',
    'splice': 'binary/splice',
    'splice_scale': 'binary/splice_scale',
    'splice_test': 'binary/splice.t',
    'SUSY': 'binary/SUSY.bz2',
    'svmguide1': 'binary/svmguide1',
    'svmguide1_test': 'binary/svmguide1.t',
    # 'svmguide3': 'binary/svmguide3',
    # 'svmguide3_test': 'binary/svmguide3.t',
    'url': 'binary/url_combined.bz2',
    'usps': 'multiclass/usps.bz2',
    'usps_test': 'multiclass/usps.t.bz2',
    'w1a': 'binary/w1a',
    'w1a_test': 'binary/w1a.t',
    'w2a': 'binary/w2a',
    'w2a_test': 'binary/w2a.t',
    'w3a': 'binary/w1a',
    'w3a_test': 'binary/w1a.t',
    'w4a': 'binary/w1a',
    'w4a_test': 'binary/w1a.t',
    'w5a': 'binary/w1a',
    'w5a_test': 'binary/w1a.t',
    'w6a': 'binary/w1a',
    'w6a_test': 'binary/w1a.t',
    'w7a': 'binary/w7a',
    'w7a_test': 'binary/w7a.t',
    'w8a': 'binary/w8a',
    'w8a_test': 'binary/w8a.t',
    'w9a': 'binary/w9a',
    'w9a_test': 'binary/w9a.t',
    'webspam': 'binary/webspam_wc_normalized_trigram.svm.bz2',
    'yeast': 'multilabel/yeast_train.svm.bz2',
    'yeast_test': 'multilabel/yeast_test.svm.bz2',
}

N_FEATURES = {
    'a1a': 123,
    'a1a_test': 123,
    'a2a': 123,
    'a2a_test': 123,
    'a3a': 123,
    'a3a_test': 123,
    'a4a': 123,
    'a4a_test': 123,
    'a5a': 123,
    'a5a_test': 123,
    'a6a': 123,
    'a6a_test': 123,
    'a7a': 123,
    'a7a_test': 123,
    'a8a': 123,
    'a8a_test': 123,
    'a9a': 123,
    'a9a_test': 123,
    'abalone': 8,
    'abalone_scale': 8,
    'aloi': 128,
    'australian': 14,
    'australian_scale': 14,
    'bibtex': 1_836,
    'bodyfat': 14,
    'breast-cancer': 10,
    'breast-cancer_scale': 10,
    'cadata': 8,
    'cifar10': 3_072,
    'cifar10_test': 3_072,
    'cod-rna': 8,
    'cod-rna_test': 8,
    'colon-cancer': 2_000,
    'connect-4': 126,
    'covtype.binary': 54,
    'covtype.multiclass': 54,
    'covtype.multiclass_scale': 54,
    'cpusmall': 12,
    'delicious': 500,
    'diabetes': 8,
    'diabetes_scale': 8,
    'dna': 180,
    'duke breast-cancer': 7_129,
    'epsilon': 2_000,
    'epsilon_test': 2_000,
    'eunite2001': 16,
    'finance': 4_272_227,
    'finance-tf-idf': 150_360,
    'fourclass': 2,
    'fourclass_scale': 2,
    'german.numer': 24,
    'german.numer_scale': 24,
    'gisette': 5_000,
    'glass': 9,
    'HIGGS': 28,
    'heart': 13,
    'heart_scale': 13,
    'housing': 13,
    'ijcnn1': 22,
    'ijcnn1_test': 22,
    'ionosphere': 34,
    'iris': 4,
    'kdda_train': 20_216_830,
    'letter': 16,
    'leukemia': 7_129,
    'leukemia_test': 7_129,
    'liver-disorders': 5,
    'liver-disorders_scale': 5,
    'liver-disorders_test': 5,
    'madelon': 500,
    'madelon_test': 500,
    'mediamill': 120,
    'mediamill_test': 120,
    'mnist': 780,
    'news20.binary': 1_355_191,
    'news20.multiclass': 62_061,
    'pendigits': 16,
    'pendigits_test': 16,
    'phishing': 68,
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
    'skin_nonskin': 3,
    'smallNORB': 18_432,
    'sonar': 60,
    'splice': 60,
    'splice_scale': 60,
    'splice_test': 60,
    'SUSY': 18,
    'svmguide1': 4,
    'svmguide1_test': 4,
    # 'svmguide3': 21,
    # 'svmguide3_test': 21,
    'url': 3_231_961,
    'usps': 7_291,
    'usps_test': 7_291,
    'w1a': 300,
    'w1a_test': 300,
    'w2a': 300,
    'w2a_test': 300,
    'w3a': 300,
    'w3a_test': 300,
    'w4a': 300,
    'w4a_test': 300,
    'w5a': 300,
    'w5a_test': 300,
    'w6a': 300,
    'w6a_test': 300,
    'w7a': 300,
    'w7a_test': 300,
    'w8a': 300,
    'w8a_test': 300,
    'w9a': 300,
    'w9a_test': 300,
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
    X_path = DATA_HOME / f"{stripped_name}_data"  # no ext to handle npy or npz
    if (replace or not y_path.exists()
        or not ((X_path.parent / (X_path.name + '.npy')).exists() or
                (X_path.parent / (X_path.name + '.npz')).exists())):
        # above, do not use .with_suffix bc of datasets like a1a.t, where the
        # method would replace the .t by .npz
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
        # if X's density is more than 0.5, store it in dense format:
        if len(X.data) >= 0.5 * X.shape[0] * X.shape[1]:
            X = X.toarray(order='F')
            np.save(X_path, X)
        else:
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
        try:
            X = sparse.load_npz(X_path.parent / (X_path.name + '.npz'))
        except FileNotFoundError:
            X = np.load(X_path.parent / (X_path.name + '.npy'))

        if multilabel:
            y = sparse.load_npz(y_path)
        else:
            y = np.load(y_path)

    return X, y


def fetch_libsvm(dataset, replace=False, normalize=False, min_nnz=0):
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
    if dataset not in NAMES:
        raise ValueError("Unsupported dataset %s. " % dataset +
                         "Supported datasets are: \n- " + '\n- '.join(NAMES))
    multilabel = NAMES[dataset].split('/')[0] == 'multilabel'
    is_regression = NAMES[dataset].split('/')[0] == 'regression'

    print("Dataset: %s" % dataset)
    X, y = _get_X_y(dataset, multilabel, replace=replace)

    # removing columns with to few non zero entries when using sparse X
    if sparse.issparse(X) and min_nnz != 0:
        X = X[:, np.diff(X.indptr) >= min_nnz]

    if normalize:
        X = preprocessing.normalize(X, axis=0)
        if is_regression:
            y -= np.mean(y)
            y /= np.std(y)

    return X, y
