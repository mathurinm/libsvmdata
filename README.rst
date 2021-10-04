|image0| |image1|

A python util to fetch datasets from the LIBSVM website.


Getting design matrix and target variable is as easy as:

::

    from libsvmdata import fetch_libsvm
    X, y = fetch_libsvm("news20.binary")


Currently supported datasets are in ``libsvmdata.supported``.


The datasets are saved in a subfolder ``libsvm`` inside ``libsvmdata.datasets.DATA_HOME``, whose value is:

- the environment variable LIBSVMDATA_HOME if it exists,

- else, the environment variable XDG_DATA_HOME if it exists,

- else, $HOME/data.



.. |image0| image:: https://github.com/mathurinm/libsvmdata/actions/workflows/build.yml/badge.svg?branch=main
   :target: https://github.com/mathurinm/libsvmdata/actions/workflows/build.yml
.. |image1| image:: https://codecov.io/gh/mathurinm/libsvmdata/branch/main/graphs/badge.svg?branch=main
   :target: https://codecov.io/gh/mathurinm/libsvmdata
