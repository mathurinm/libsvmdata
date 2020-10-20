import os
from setuptools.command.build_ext import build_ext
from setuptools import dist, setup, Extension, find_packages

descr = 'Fetcher for LIBSVM datasets'

version = None
with open(os.path.join('libsvmdata', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'libsvmdata'
DESCRIPTION = descr
MAINTAINER = 'Mathurin Massias'
MAINTAINER_EMAIL = 'mathurin.massias@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mathurinm/libsvmdata.git'
VERSION = version

setup(name='libsvmdata',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.rst').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      download_url=DOWNLOAD_URL,
      install_requires=['download', 'numpy>=1.12', 'scikit-learn', 'scipy'],
      packages=find_packages(),
      )
