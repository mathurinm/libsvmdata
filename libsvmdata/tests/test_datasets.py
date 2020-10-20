from libsvmdata import fetch_libsvm


def test_news20():
    X, y = fetch_libsvm('news20')
    np.testing.assert_equal(X.shape[0], y.shape[0])
