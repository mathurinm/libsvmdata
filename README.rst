|image0| |image1|

A python util to fetch datasets from different databases.

Currently supported databases are:

- LIBSVM (libsvm_)

Currently supported datasets are in ``libsvmdata.SUPPORTED`` and can be displayed as: 

::

   from libsvmdata import print_supported_datasets
   print_supported_datasets()

Getting design matrix and target variable is as easy as:

::

   from libsvmdata import fetch_dataset
   X, y = fetch_dataset("news20.binary")

There is no need to specify the database name.

Files are saved under ``DATA_HOME/<database_name>``, where the value of ``DATA_HOME`` is:

- the environment variable ``LIBSVMDATA_HOME`` if it exists,

- else, the environment variable ``XDG_DATA_HOME`` if it exists,

- else, ``$HOME/data``.



.. |image0| image:: https://github.com/mathurinm/libsvmdata/actions/workflows/build.yml/badge.svg?branch=main
   :target: https://github.com/mathurinm/libsvmdata/actions/workflows/build.yml
.. |image1| image:: https://codecov.io/gh/mathurinm/libsvmdata/branch/main/graphs/badge.svg?branch=main
   :target: https://codecov.io/gh/mathurinm/libsvmdata
.. _libsvm: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
.. _breheny: https://myweb.uiowa.edu/pbreheny/7240/s21/data.html