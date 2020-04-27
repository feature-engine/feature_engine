from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "feature_engine"
DESCRIPTION = "Feature engineering package with Scikit-learn's fit transform functionality"
URL = "http://github.com/solegalli/feature_engine"
EMAIL = "solegalli@protonmail.com"
AUTHOR = "Soledad Galli"
REQUIRES_PYTHON = ">=3.6.0"

# description
with open("README.md", "r") as fh:
    long_description = fh.read()


# Packages required for this module to be executed
def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'feature_engine'
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

setup(name=NAME,
      version=about["__version__"],
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type="text/markdown",
      url=URL,
      author=AUTHOR,
      author_email=EMAIL,
      python_requires=REQUIRES_PYTHON,
      packages=find_packages(exclude=("tests",)),
      package_data={"feature_engine": ["VERSION"]},
      license='BSD 3 clause',
      install_requires=list_reqs(),
      include_package_data=True,
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
