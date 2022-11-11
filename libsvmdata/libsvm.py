import numpy as np
from bz2 import BZ2Decompressor
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from libsvmdata.core import _get_data_home, AbstractDataset

LIBSVM_DATA_HOME = _get_data_home("libsvm")
LIBSVM_BASE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"


class LibsvmDataset(AbstractDataset):
    def __init__(self, dataset_name, dataset_file, task_name, n_features):

        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.dataset_dir = LIBSVM_DATA_HOME
        self.dataset_url = "/".join([LIBSVM_BASE_URL, task_name, dataset_file])
        self.task_name = task_name
        self.n_features = n_features

    def _load_file_and_save_data(self, raw_dataset_path, ext_dataset_path):

        # Handle .bz2 compressed datasets
        if str(raw_dataset_path).endswith(".bz2"):
            tmp_dataset_path = raw_dataset_path.with_suffix("")
            decompressor = BZ2Decompressor()
            # TODO : why not using bz2.decompress(raw_dataset_path) only ?
            with open(tmp_dataset_path, "wb") as f, open(
                raw_dataset_path, "rb"
            ) as g:
                for data in iter(lambda: g.read(100 * 1024), b""):
                    f.write(decompressor.decompress(data))
            raw_dataset_path.unlink()
            raw_dataset_path = tmp_dataset_path

        with open(raw_dataset_path, "rb") as file:
            X, y = load_svmlight_file(
                file,
                n_features=self.n_features,
                multilabel=(self.task_name == "multilabel"),
            )
        raw_dataset_path.unlink()

        X_path = str(ext_dataset_path) + "_X"
        if len(X.data) >= 0.5 * X.shape[0] * X.shape[1]:
            X = X.toarray(order="F")
            np.save(X_path, X)
        else:
            X = sparse.csc_matrix(X)
            X.sort_indices()
            sparse.save_npz(X_path, X)

        y_path = str(ext_dataset_path) + "_y"
        if self.task_name == "multilabel":
            indices = np.array([lab for labels in y for lab in labels])
            indptr = np.cumsum([0] + [len(labels) for labels in y])
            data = np.ones_like(indices)
            y = sparse.csr_matrix((data, indices, indptr))
            sparse.save_npz(y_path, y)
        else:
            np.save(y_path, y)

        return X, y


DATASETS = [
    LibsvmDataset("a1a", "a1a", "binary", 123),
    LibsvmDataset("a1a_test", "a1a.t", "binary", 123),
    LibsvmDataset("a2a", "a2a", "binary", 123),
    LibsvmDataset("a2a_test", "a2a.t", "binary", 123),
    LibsvmDataset("a3a", "a3a", "binary", 123),
    LibsvmDataset("a3a_test", "a3a.t", "binary", 123),
    LibsvmDataset("a4a", "a4a", "binary", 123),
    LibsvmDataset("a4a_test", "a4a.t", "binary", 123),
    LibsvmDataset("a5a", "a5a", "binary", 123),
    LibsvmDataset("a5a_test", "a5a.t", "binary", 123),
    LibsvmDataset("a6a", "a6a", "binary", 123),
    LibsvmDataset("a6a_test", "a6a.t", "binary", 123),
    LibsvmDataset("a7a", "a7a", "binary", 123),
    LibsvmDataset("a7a_test", "a7a.t", "binary", 123),
    LibsvmDataset("a8a", "a8a", "binary", 123),
    LibsvmDataset("a8a_test", "a8a.t", "binary", 123),
    LibsvmDataset("a9a", "a9a", "binary", 123),
    LibsvmDataset("a9a_test", "a9a.t", "binary", 123),
    LibsvmDataset("abalone", "abalone", "regression", 8),
    LibsvmDataset("abalone_scale", "abalone_scale", "regression", 8),
    LibsvmDataset("aloi", "aloi.bz2", "multiclass", 128),
    LibsvmDataset("australian", "australian", "binary", 14),
    LibsvmDataset("australian_scale", "australian_scale", "binary", 14),
    LibsvmDataset("bibtex", "bibtex.bz2", "multilabel", 1836),
    LibsvmDataset("bodyfat", "bodyfat", "regression", 14),
    LibsvmDataset("breast-cancer", "breast-cancer", "binary", 10),
    LibsvmDataset("breast-cancer_scale", "breast-cancer_scale", "binary", 10),
    LibsvmDataset("cadata", "cadata", "regression", 8),
    LibsvmDataset("cifar10", "cifar10.bz2", "multiclass", 3072),
    LibsvmDataset("cifar10_test", "cifar10.t.bz2", "multiclass", 3072),
    LibsvmDataset("cod-rna", "cod-rna", "binary", 8),
    LibsvmDataset("cod-rna_test", "cod-rna.t", "binary", 8),
    LibsvmDataset("colon-cancer", "colon-cancer.bz2", "binary", 2000),
    LibsvmDataset("connect-4", "connect-4", "multiclass", 126),
    LibsvmDataset("covtype.binary", "covtype.libsvm.binary.bz2", "binary", 54),
    LibsvmDataset("covtype.multiclass", "covtype.bz2", "multiclass", 54),
    LibsvmDataset(
        "covtype.multiclass_scale", "covtype.scale01.bz2", "multiclass", 54
    ),
    LibsvmDataset("cpusmall", "cpusmall", "regression", 12),
    LibsvmDataset("delicious", "delicious.bz2", "multilabel", 500),
    LibsvmDataset("diabetes", "diabetes", "binary", 8),
    LibsvmDataset("diabetes_scale", "diabetes_scale", "binary", 8),
    LibsvmDataset("dna", "dna.scale", "multiclass", 180),
    LibsvmDataset("duke breast-cancer", "duke.bz2", "binary", 7129),
    LibsvmDataset("epsilon", "epsilon_normalized.bz2", "binary", 2000),
    LibsvmDataset("epsilon_test", "epsilon_normalized.t.bz2", "binary", 2000),
    LibsvmDataset("eunite2001", "eunite2001", "regression", 16),
    LibsvmDataset("finance", "log1p.E2006.train.bz2", "regression", 4272227),
    LibsvmDataset("finance-tf-idf", "E2006.train.bz2", "regression", 150360),
    LibsvmDataset("fourclass", "fourclass", "binary", 2),
    LibsvmDataset("fourclass_scale", "fourclass_scale", "binary", 2),
    LibsvmDataset("german.numer", "german.numer", "binary", 24),
    LibsvmDataset("german.numer_scale", "german.numer_scale", "binary", 24),
    LibsvmDataset("gisette", "gisette_scale.bz2", "binary", 5000),
    LibsvmDataset("glass", "glass.scale", "multiclass", 9),
    LibsvmDataset("heart", "heart", "binary", 13),
    LibsvmDataset("heart_scale", "heart_scale", "binary", 13),
    LibsvmDataset("HIGGS", "HIGGS.bz2", "binary", 28),
    LibsvmDataset("housing", "housing", "regression", 13),
    LibsvmDataset("ijcnn1", "ijcnn1.bz2", "binary", 22),
    LibsvmDataset("ijcnn1_test", "ijcnn1.t.bz2", "binary", 22),
    LibsvmDataset("ionosphere", "ionosphere_scale", "binary", 34),
    LibsvmDataset("iris", "iris.scale", "multiclass", 4),
    LibsvmDataset("kdda_train", "kdda.bz2", "binary", 20216830),
    LibsvmDataset("letter", "letter.scale", "multiclass", 16),
    LibsvmDataset("leukemia", "leu.bz2", "binary", 7129),
    LibsvmDataset("leukemia_test", "leu.t.bz2", "binary", 7129),
    LibsvmDataset("liver-disorders", "liver-disorders", "binary", 5),
    LibsvmDataset(
        "liver-disorders_scale", "liver-disorders_scale", "binary", 5
    ),
    LibsvmDataset("liver-disorders_test", "liver-disorders.t", "binary", 5),
    LibsvmDataset("madelon", "madelon", "binary", 500),
    LibsvmDataset("madelon_test", "madelon.t", "binary", 500),
    LibsvmDataset(
        "mediamill", "mediamill/train-exp1.svm.bz2", "multilabel", 120
    ),
    LibsvmDataset(
        "mediamill_test", "mediamill/test-exp1.svm.bz2", "multilabel", 120
    ),
    LibsvmDataset("mnist", "mnist.bz2", "multiclass", 780),
    LibsvmDataset("news20.binary", "news20.binary.bz2", "binary", 1355191),
    LibsvmDataset("news20.multiclass", "news20.bz2", "multiclass", 62061),
    LibsvmDataset("pendigits", "pendigits", "multiclass", 16),
    LibsvmDataset("pendigits_test", "pendigits.t", "multiclass", 16),
    LibsvmDataset("phishing", "phishing", "binary", 68),
    LibsvmDataset("rcv1.binary", "rcv1_train.binary.bz2", "binary", 47236),
    LibsvmDataset("rcv1.binary_test", "rcv1_test.binary.bz2", "binary", 47236),
    LibsvmDataset(
        "rcv1.multiclass", "rcv1_train.multiclass.bz2", "multiclass", 47236
    ),
    LibsvmDataset(
        "rcv1.multiclass_test", "rcv1_test.multiclass.bz2", "multiclass", 47236
    ),
    LibsvmDataset(
        "rcv1_topics_test", "rcv1_topics_test_2.svm.bz2", "multilabel", 47236
    ),
    LibsvmDataset("real-sim", "real-sim.bz2", "binary", 20958),
    LibsvmDataset(
        "scene-classification", "scene_train.bz2", "multilabel", 294
    ),
    LibsvmDataset(
        "scene-classification_test", "scene_test.bz2", "multilabel", 294
    ),
    LibsvmDataset(
        "sector.scale", "sector/sector.scale.bz2", "multiclass", 55197
    ),
    LibsvmDataset(
        "sector.scale_test", "sector/sector.t.scale.bz2", "multiclass", 55197
    ),
    LibsvmDataset("sector", "sector/sector.bz2", "multiclass", 55197),
    LibsvmDataset("sector_test", "sector/sector.t.bz2", "multiclass", 55197),
    LibsvmDataset("sensit", "vehicle/combined.bz2", "multiclass", 100),
    LibsvmDataset(
        "siam-competition2007", "tmc2007_train.svm.bz2", "multilabel", 30438
    ),
    LibsvmDataset(
        "siam-competition2007_test",
        "tmc2007_test.svm.bz2",
        "multilabel",
        30438,
    ),
    LibsvmDataset("skin_nonskin", "skin_nonskin", "binary", 3),
    LibsvmDataset("smallNORB", "smallNORB.bz2", "multiclass", 18432),
    LibsvmDataset("sonar", "sonar_scale", "binary", 60),
    LibsvmDataset("splice", "splice", "binary", 60),
    LibsvmDataset("splice_scale", "splice_scale", "binary", 60),
    LibsvmDataset("splice_test", "splice.t", "binary", 60),
    LibsvmDataset("SUSY", "SUSY.bz2", "binary", 18),
    LibsvmDataset("svmguide1", "svmguide1", "binary", 4),
    LibsvmDataset("svmguide1_test", "svmguide1.t", "binary", 4),
    LibsvmDataset("url", "url_combined.bz2", "binary", 3231961),
    LibsvmDataset("usps", "usps.bz2", "multiclass", 7291),
    LibsvmDataset("usps_test", "usps.t.bz2", "multiclass", 7291),
    LibsvmDataset("w1a", "w1a", "binary", 300),
    LibsvmDataset("w1a_test", "w1a.t", "binary", 300),
    LibsvmDataset("w2a", "w2a", "binary", 300),
    LibsvmDataset("w2a_test", "w2a.t", "binary", 300),
    LibsvmDataset("w3a", "w1a", "binary", 300),
    LibsvmDataset("w3a_test", "w1a.t", "binary", 300),
    LibsvmDataset("w4a", "w1a", "binary", 300),
    LibsvmDataset("w4a_test", "w1a.t", "binary", 300),
    LibsvmDataset("w5a", "w1a", "binary", 300),
    LibsvmDataset("w5a_test", "w1a.t", "binary", 300),
    LibsvmDataset("w6a", "w1a", "binary", 300),
    LibsvmDataset("w6a_test", "w1a.t", "binary", 300),
    LibsvmDataset("w7a", "w7a", "binary", 300),
    LibsvmDataset("w7a_test", "w7a.t", "binary", 300),
    LibsvmDataset("w8a", "w8a", "binary", 300),
    LibsvmDataset("w8a_test", "w8a.t", "binary", 300),
    LibsvmDataset("w9a", "w9a", "binary", 300),
    LibsvmDataset("w9a_test", "w9a.t", "binary", 300),
    LibsvmDataset(
        "webspam", "webspam_wc_normalized_trigram.svm.bz2", "binary", 16609143
    ),
    LibsvmDataset(
        "YearPredictionMSD", "YearPredictionMSD.bz2", "regression", 90
    ),
    LibsvmDataset(
        "YearPredictionMSD_test", "YearPredictionMSD.t.bz2", "regression", 90
    ),
    LibsvmDataset("yeast", "yeast_train.svm.bz2", "multilabel", 103),
    LibsvmDataset("yeast_test", "yeast_test.svm.bz2", "multilabel", 103),
]
