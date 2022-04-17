# Feature Engine

![PythonVersion](https://img.shields.io/badge/python-3.6%20|3.7%20|%203.8%20|%203.9-success)
[![License https://github.com/feature-engine/feature_engine/blob/master/LICENSE.md](https://img.shields.io/badge/license-BSD-success.svg)](https://github.com/feature-engine/feature_engine/blob/master/LICENSE.md)
[![PyPI version](https://badge.fury.io/py/feature-engine.svg)](https://badge.fury.io/py/feature-engine)
[![Conda https://anaconda.org/conda-forge/feature_engine](https://anaconda.org/conda-forge/feature_engine/badges/installer/conda.svg)](https://anaconda.org/conda-forge/feature_engine)
[![CircleCI https://app.circleci.com/pipelines/github/feature-engine/feature_engine](https://img.shields.io/circleci/build/github/feature-engine/feature_engine/main)](https://app.circleci.com/pipelines/github/feature-engine/feature_engine?)
[![Documentation Status https://feature-engine.readthedocs.io/en/latest/index.html](https://readthedocs.org/projects/feature-engine/badge/?version=latest)](https://feature-engine.readthedocs.io/en/latest/index.html)
[![Join the chat at https://gitter.im/feature_engine/community](https://badges.gitter.im/feature_engine/community.svg)](https://gitter.im/feature_engine/community)
[![Sponsorship https://www.trainindata.com/](https://img.shields.io/badge/Powered%20By-TrainInData-orange.svg)](https://www.trainindata.com/)
[![Downloads](https://pepy.tech/badge/feature-engine)](https://pepy.tech/project/feature-engine)
[![Downloads](https://pepy.tech/badge/feature-engine/month)](https://pepy.tech/project/feature-engine)
[![DOI](https://zenodo.org/badge/163630824.svg)](https://zenodo.org/badge/latestdoi/163630824)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03642/status.svg)](https://doi.org/10.21105/joss.03642)


[<img src="https://github.com/feature-engine/feature_engine/blob/main/docs/images/logo/FeatureEngine.png" width="248">](http://feature-engine.readthedocs.io)

Feature-engine is a Python library with multiple transformers to engineer and select features for use in machine learning models. 
Feature-engine's transformers follow Scikit-learn's functionality with fit() and transform() methods to learn the 
transforming parameters from the data and then transform it.


## Feature-engine features in the following resources

* [Feature Engineering for Machine Learning, Online Course](https://courses.trainindata.com/p/feature-engineering-for-machine-learning)

* [Feature Selection for Machine Learning, Online Course](https://courses.trainindata.com/p/feature-selection-for-machine-learning)

* [Feature Engineering for Time Series Forecasting, Online Course](https://www.courses.trainindata.com/p/feature-engineering-for-forecasting)

* [Deployment of Machine Learning Models, Online Course](https://www.udemy.com/course/deployment-of-machine-learning-models/?referralCode=D4FE5EA129FFD203CFF4)

* [Python Feature Engineering Cookbook](https://packt.link/python)


## Blogs about Feature-engine

* [Feature-engine: A new open-source Python package for feature engineering](https://trainindata.medium.com/feature-engine-a-new-open-source-python-package-for-feature-engineering-29a0ab88ea7c)

* [Practical Code Implementations of Feature Engineering for Machine Learning with Python](https://towardsdatascience.com/practical-code-implementations-of-feature-engineering-for-machine-learning-with-python-f13b953d4bcd)


## En Español

* [Ingeniería de variables para machine learning, Curso Online](https://www.udemy.com/course/ingenieria-de-variables-para-machine-learning/?referralCode=CE398C784F17BD87482C)

* [Ingeniería de variables, MachinLenin, charla online](https://www.youtube.com/watch?v=NhCxOOoFXds)


## Documentation

* [Documentation](http://feature-engine.readthedocs.io)


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
* PRatioEncoder
* RareLabelEncoder
* DecisionTreeEncoder

### Discretisation methods
* EqualFrequencyDiscretiser
* EqualWidthDiscretiser
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

### Datetime
 * DatetimeFeatures
 
### Time Series
 * LagFeatures
 * WindowFeatures
 * ExpandingWindowFeatures
 
### Preprocessing
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
or in the [documentation](http://feature-engine.readthedocs.io).

## Contribute

Details about how to contribute can be found in the [Contribute Page](https://feature-engine.readthedocs.io/en/latest/contribute/index.html)

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

BSD 3-Clause

## Sponsor us

[Sponsor us](https://github.com/sponsors/feature-engine) and support further our 
mission to democratize machine learning and programming tools through open-source 
software.
