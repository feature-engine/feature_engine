from pathlib import Path

from setuptools import find_packages, setup

# description
with open("README.md", "r") as fh:
    long_description = fh.read()


# Packages required for this module to be executed
def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


def list_test_reqs(fname='test_requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'feature_engine'
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

setup(name='feature_engine',
      version=about["__version__"],
      description="Feature engineering package with Scikit-klearn's fit transform functionality",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/solegalli/feature_engine',
      author='Soledad Galli',
      author_email='solegalli@protonmail.com',
      packages=['feature_engine'],
      license='BSD 3 clause',
      install_requires=list_reqs(),
      tests_require=list_test_reqs(),
      classifiers=[
          # Trove classifiers
          # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
      ],
      zip_safe=False)
