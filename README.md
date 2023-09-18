# Feature Engine

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/feature_engine?logo=Python)](https://pypi.org/project/feature-engine/)
[![GitHub](https://img.shields.io/github/license/feature-engine/feature_engine)](https://github.com/feature-engine/feature_engine/blob/master/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/feature_engine?logo=PyPI)](https://pypi.org/project/feature-engine)
[![Conda](https://img.shields.io/conda/v/conda-forge/feature_engine?logo=Anaconda)](https://anaconda.org/conda-forge/feature_engine)
[![CircleCI](https://img.shields.io/circleci/build/github/feature-engine/feature_engine/main?logo=CircleCI)](https://app.circleci.com/pipelines/github/feature-engine/feature_engine)
[![Codecov](https://img.shields.io/codecov/c/github/feature-engine/feature_engine?logo=CodeCov&token=ZBKKSN6ERL)](https://codecov.io/github/feature-engine/feature_engine)
[![Read the Docs](https://img.shields.io/readthedocs/feature_engine?logo=readthedocs)](https://feature-engine.readthedocs.io/en/latest/index.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub contributors](https://img.shields.io/github/contributors/feature-engine/feature_engine?logo=GitHub)](https://github.com/feature-engine/feature_engine/graphs/contributors)
[![Gitter](https://img.shields.io/gitter/room/feature-engine/feaure_engine?logo=Gitter)](https://gitter.im/feature_engine/community)
[![Monthly Downloads](https://img.shields.io/pypi/dm/feature-engine)](https://img.shields.io/pypi/dm/feature-engine)
[![DOI](https://zenodo.org/badge/163630824.svg)](https://zenodo.org/badge/latestdoi/163630824)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.03642/status.svg)](https://doi.org/10.21105/joss.03642)
[![first-timers-only](https://img.shields.io/badge/first--timers--only-friendly-blue.svg?style=flat)](https://www.firsttimersonly.com/)
[![Sponsorship](https://img.shields.io/badge/Powered%20By-TrainInData-orange.svg)](https://www.trainindata.com/)

<div align="center">

[![feature-engine logo](https://raw.githubusercontent.com/feature-engine/feature_engine/main/docs/images/logo/FeatureEngine.png)](http://feature-engine.readthedocs.io)

</div>

Feature-engine is a Python library with multiple transformers to engineer and select features for use in machine learning models. 
Feature-engine's transformers follow Scikit-learn's functionality with fit() and transform() methods to learn the 
transforming parameters from the data and then transform it.


## Feature-engine features in the following resources

* [Feature Engineering for Machine Learning, Online Course](https://www.trainindata.com/p/feature-engineering-for-machine-learning)

* [Feature Selection for Machine Learning, Online Course](https://www.trainindata.com/p/feature-selection-for-machine-learning)

* [Feature Engineering for Time Series Forecasting, Online Course](https://www.trainindata.com/p/feature-engineering-for-forecasting)

* [Python Feature Engineering Cookbook](https://packt.link/0ewSo)

* [Feature Selection in Machine Learning with Python Book](https://leanpub.com/feature-selection-in-machine-learning)


## Blogs about Feature-engine

* [Feature-engine: A new open-source Python package for feature engineering](https://trainindata.medium.com/feature-engine-a-new-open-source-python-package-for-feature-engineering-29a0ab88ea7c)

* [Practical Code Implementations of Feature Engineering for Machine Learning with Python](https://towardsdatascience.com/practical-code-implementations-of-feature-engineering-for-machine-learning-with-python-f13b953d4bcd)


## Documentation

* [Documentation](https://feature-engine.trainindata.com)


## Current Feature-engine's transformers include functionality for:

* Missing Data Imputation
* Categorical Encoding
* Discretisation
* Outlier Capping or Removal
* Variable Transformation
* Variable Creation
* Variable Selection
* Datetime Features
* Time Series
* Preprocessing
* Scikit-learn Wrappers

### Imputation Methods
* MeanMedianImputer
* RandomSampleImputer
* EndTailImputer
* AddMissingIndicator
* CategoricalImputer
* ArbitraryNumberImputer
* DropMissingData

### Encoding Methods
* OneHotEncoder
* OrdinalEncoder
* CountFrequencyEncoder
* MeanEncoder
* WoEEncoder
* RareLabelEncoder
* DecisionTreeEncoder
* StringSimilarityEncoder

### Discretisation methods
* EqualFrequencyDiscretiser
* EqualWidthDiscretiser
* GeometricWidthDiscretiser
* DecisionTreeDiscretiser
* ArbitraryDiscreriser

### Outlier Handling methods
* Winsorizer
* ArbitraryOutlierCapper
* OutlierTrimmer

### Variable Transformation methods
* LogTransformer
* LogCpTransformer
* ReciprocalTransformer
* ArcsinTransformer
* PowerTransformer
* BoxCoxTransformer
* YeoJohnsonTransformer

### Variable Creation:
 * MathFeatures
 * RelativeFeatures
 * CyclicalFeatures

### Feature Selection:
 * DropFeatures
 * DropConstantFeatures
 * DropDuplicateFeatures
 * DropCorrelatedFeatures
 * SmartCorrelationSelection
 * ShuffleFeaturesSelector
 * SelectBySingleFeaturePerformance
 * SelectByTargetMeanPerformance
 * RecursiveFeatureElimination
 * RecursiveFeatureAddition
 * DropHighPSIFeatures
 * SelectByInformationValue
 * ProbeFeatureSelection

### Datetime
 * DatetimeFeatures
 * DatetimeSubtraction
 
### Time Series
 * LagFeatures
 * WindowFeatures
 * ExpandingWindowFeatures
 
### Preprocessing
 * MatchCategories
 * MatchVariables
 
### Wrappers:
 * SklearnTransformerWrapper

## Installation

From PyPI using pip:

```
pip install feature_engine
```

From Anaconda:

```
conda install -c conda-forge feature_engine
```

Or simply clone it:

```
git clone https://github.com/feature-engine/feature_engine.git
```

## Example Usage

```python
>>> import pandas as pd
>>> from feature_engine.encoding import RareLabelEncoder

>>> data = {'var_A': ['A'] * 10 + ['B'] * 10 + ['C'] * 2 + ['D'] * 1}
>>> data = pd.DataFrame(data)
>>> data['var_A'].value_counts()
```

```
Out[1]:
A    10
B    10
C     2
D     1
Name: var_A, dtype: int64
```
    
```python 
>>> rare_encoder = RareLabelEncoder(tol=0.10, n_categories=3)
>>> data_encoded = rare_encoder.fit_transform(data)
>>> data_encoded['var_A'].value_counts()
```

```
Out[2]:
A       10
B       10
Rare     3
Name: var_A, dtype: int64
```

Find more examples in our [Jupyter Notebook Gallery](https://nbviewer.org/github/feature-engine/feature-engine-examples/tree/main/) 
or in the [documentation](https://feature-engine.trainindata.com).

## Contribute

Details about how to contribute can be found in the [Contribute Page](https://feature-engine.trainindata.com/en/latest/contribute/index.html)

Briefly:

- Fork the repo
- Clone your fork into your local computer: ``git clone https://github.com/<YOURUSERNAME>/feature_engine.git``
- navigate into the repo folder ``cd feature_engine``
- Install Feature-engine as a developer: ``pip install -e .``
- Optional: Create and activate a virtual environment with any tool of choice
- Install Feature-engine dependencies: ``pip install -r requirements.txt`` and ``pip install -r test_requirements.txt``
- Create a feature branch with a meaningful name for your feature: ``git checkout -b myfeaturebranch``
- Develop your feature, tests and documentation
- Make sure the tests pass
- Make a PR

Thank you!!


### Documentation

Feature-engine documentation is built using [Sphinx](https://www.sphinx-doc.org) and is hosted on [Read the Docs](https://readthedocs.org/).

To build the documentation make sure you have the dependencies installed: from the root directory: ``pip install -r docs/requirements.txt``.

Now you can build the docs using: ``sphinx-build -b html docs build``


## License

The content of this repository is licensed under a [BSD 3-Clause license](https://github.com/feature-engine/feature_engine/blob/main/LICENSE.md).

## Sponsor us

[Sponsor us](https://github.com/sponsors/feature-engine) and support further our 
mission to democratize machine learning and programming tools through open-source 
software.
