# Feature Engine

![Python 3.6](https://img.shields.io/badge/python-3.6-success.svg)
![Python 3.7](https://img.shields.io/badge/python-3.7-success.svg)
![Python 3.8](https://img.shields.io/badge/python-3.8-success.svg)
![License](https://img.shields.io/badge/license-BSD-success.svg)
![CircleCI](https://img.shields.io/circleci/build/github/solegalli/feature_engine/master.svg?token=5a1c2accc2c97450e52d2cb1b47c333ab495d2c2)
![Documentation Status](https://readthedocs.org/projects/feature-engine/badge/?version=latest)


Feature-engine is a Python library with multiple transformers to engineer features for use in machine learning models. 
Feature-engine's transformers follow scikit-learn's functionality with fit() and transform() methods to first learn the 
transforming parameters from data and then transform the data.


## Feature-engine features in the following resources:

* [Feature Engineering for Machine Learning, Online Course](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=FEATENGREPO)

* [Python Feature Engineering Cookbook](https://www.packtpub.com/data/python-feature-engineering-cookbook)

## Blogs about Feature-engine:

* [Feature-engine: A new open-source Python package for feature engineering](https://www.trainindatablog.com/feature-engine-a-new-open-source-python-package-for-feature-engineering)

* [Practical Code Implementations of Feature Engineering for Machine Learning with Python](https://www.trainindatablog.com/practical-code-implementations-of-feature-engineering-for-machine-learning-with-python)

## Documentation

* [Documentation](http://feature-engine.readthedocs.io)
* [Home page](https://www.trainindata.com/feature-engine)

## En Español:

* [Ingeniería de variables para machine learning, Curso Online](https://www.udemy.com/course/ingenieria-de-variables-para-machine-learning/?referralCode=CE398C784F17BD87482C)

* [Ingeniería de variables, MachinLenin, charla online](https://www.youtube.com/watch?v=NhCxOOoFXds)

More resources will be added as they appear online!


## Current Feature-engine's transformers include functionality for:
* Missing Data Imputation
* Categorical Variable Encoding
* Outlier Capping or Removal
* Discretisation
* Numerical Variable Transformation
* Scikit-learn Wrappers
* Variable Combination
* Variable Selection

### Imputing Methods
* MeanMedianImputer
* RandomSampleImputer
* EndTailImputer
* AddNaNBinaryImputer
* CategoricalImputer
* ArbitraryNumberImputer

### Encoding Methods
* OneHotEncoder
* OrdinalEncoder
* CountFrequencyEncoder
* MeanEncoder
* WoEEncoder
* PRatioEncoder
* RareLabelEncoder
* DecisionTreeEncoder

### Outlier Handling methods
* Winsorizer
* ArbitraryOutlierCapper
* OutlierTrimmer

### Discretisation methods
* EqualFrequencyDiscretiser
* EqualWidthDiscretiser
* DecisionTreeDiscretiser
* UserInputDiscreriser

### Variable Transformation methods
* LogTransformer
* ReciprocalTransformer
* PowerTransformer
* BoxCoxTransformer
* YeoJohnsonTransformer

### Scikit-learn Wrapper:
 * SklearnTransformerWrapper

### Variable Combinations:
 * MathematicalCombination
 * CombineWithReferenceFeature

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


## Installing

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
git clone https://github.com/solegalli/feature_engine.git
```

### Usage

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

See more usage examples in the Jupyter Notebooks in the **example** folder of this repository, or in the [documentation](http://feature-engine.readthedocs.io).

## Contributing

Details about how to contribute can be found in the [Contributing Page](https://feature-engine.readthedocs.io/en/latest/contributing/index.html)

In short:

### Local Setup Steps
- Fork the repo
- Clone your fork into your local computer: ``git clone https://github.com/<YOURUSERNAME>/feature_engine.git``
- cd into the repo ``cd feature_engine``
- Install as a developer: ``pip install -e .``
- Create and activate a virtual environment with any tool of choice
- Install the dependencies as explained in the [Contributing Page](https://feature-engine.readthedocs.io/en/latest/contributing/index.html)
- Create a feature branch with a meaningful name for your feature: ``git checkout -b myfeaturebranch``
- Develop your feature, tests and documentation
- Make sure the tests pass
- Make a PR

Thank you!!

### Opening Pull Requests
PR's are welcome! Please make sure the CI tests pass on your branch.

### Tests

We prefer tox. In your environment:

- Run `pip install tox`
- cd into the root directory of the repo: ``cd feature_engine``
- Run `tox` 

If the tests pass, the code is functional.

You can also run the tests in your environment (without tox). For guidelines on how to do so, check the [Contributing Page](https://feature-engine.readthedocs.io/en/latest/contributing/index.html).


### Documentation

Feature-engine documentation is built using [Sphinx](https://www.sphinx-doc.org) and is hosted on [Read the Docs](https://readthedocs.org/).

To build the documentation make sure you have the dependencies installed. From the root directory: ``pip install -r docs/requirements.txt``.

Now you can build the docs: ``sphinx-build -b html docs build``


## License

BSD 3-Clause


## References

Many of the engineering and encoding functionalities are inspired by this [series of articles from the 2009 KDD Competition](http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf).
