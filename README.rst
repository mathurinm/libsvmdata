|image0| |image1|

A python util to fetch datasets from the LIBSVM website.


Getting design matrix is as easy as:

::

    from libsvmdata import fetch_libsvm
    X, y = fetch_libsvm("news20")


Currently supported datasets are in `libsvmdata.datasets.NAMES.keys()`.


.. |image0| image:: https://travis-ci.com/mathurinm/libsvmdata.svg?branch=master
   :target: https://travis-ci.com/mathurinm/libsvmdata/
.. |image1| image:: https://codecov.io/gh/mathurinm/libsvmdata/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/mathurinm/libsvmdata
