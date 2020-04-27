# Feature Engine

![Python 3.6](https://img.shields.io/badge/python-3.6-success.svg)
![Python 3.7](https://img.shields.io/badge/python-3.7-success.svg)
![Python 3.8](https://img.shields.io/badge/python-3.8-success.svg)
![License](https://img.shields.io/badge/license-BSD-success.svg)
![CircleCI](https://img.shields.io/circleci/build/github/solegalli/feature_engine/master.svg?token=5a1c2accc2c97450e52d2cb1b47c333ab495d2c2)
![Documentation Status](https://readthedocs.org/projects/feature-engine/badge/?version=latest)


Feature-engine is a Python library with multiple transformers to engineer features for use in machine learning models. Feature-engine's transformers follow Scikit-learn functionality with fit() and transform() methods to first learn the transforming paramenters from data and then transform the data.


## Feature-engine features in the following resources:

[Feature Engineering for Machine Learning, Online Course](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=FEATENGREPO).
[Python Feature Engineering Cookbook](https://www.packtpub.com/data/python-feature-engineering-cookbook)


## Documentation

* Documentation: http://feature-engine.readthedocs.io
* Home page: https://www.trainindata.com/feature-engine


## Current Feature-engine's transformers include functionality for:

* Missing data imputation
* Categorical variable encoding
* Outlier removal
* Discretisation
* Numerical Variable Transformation

### Imputing Methods

* MeanMedianImputer
* RandomSampleImputer
* EndTailImputer
* AddNaNBinaryImputer
* CategoricalVariableImputer
* FrequentCategoryImputer
* ArbitraryNumberImputer

### Encoding Methods
* CountFrequencyCategoricalEncoder
* OrdinalCategoricalEncoder 
* MeanCategoricalEncoder
* WoERatioCategoricalEncoder
* OneHotCategoricalEncoder
* RareLabelCategoricalEncoder

### Outlier Handling methods
* Winsorizer
* ArbitraryOutlierCapper
* OutlierTrimmer

### Discretisation methods
* EqualFrequencyDiscretiser
* EqualWidthDiscretiser
* DecisionTreeDiscretiser

### Variable Transformation methods
* LogTransformer
* ReciprocalTransformer
* PowerTransformer
* BoxCoxTransformer
* YeoJohnsonTransformer


### Installing

```
pip install feature_engine
```
or

```
git clone https://github.com/solegalli/feature_engine.git
```

### Usage

```
from feature_engine.categorical_encoders import RareLabelEncoder

rare_encoder = RareLabelEncoder(tol = 0.05, n_categories=5)
rare_encoder.fit(data, variables = ['Cabin', 'Age'])
data_encoded = rare_encoder.transform(data)
```

See more usage examples in the jupyter notebooks in the **example** folder of this repository, or in the documentation: http://feature-engine.readthedocs.io

## Contributing

### Local Setup Steps
- Clone the repo and cd into it
- Run `pip install tox`
- Run `tox` if the tests pass, your local setup is complete

### Opening Pull Requests
PR's are welcome! Please make sure the CI tests pass on your branch.

## License

BSD 3-Clause

## Authors

* **Soledad Galli** - *Initial work* - [Feature Engineering for Machine Learning, Online Course](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=FEATENGREPO).


### References

Many of the engineering and encoding functionality is inspired by this [series of articles from the 2009 KDD competition](http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf).

To learn more about the rationale, functionality, pros and cos of each imputer, encoder and transformer, refer to the [Feature Engineering for Machine Learning, Online Course](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=FEATENGREPO)

For a summary of the methods check this [presentation](https://speakerdeck.com/solegalli/engineering-and-selecting-features-for-machine-learning) and this [article](https://www.trainindata.com/post/feature-engineering-comprehensive-overview)

To stay alert of latest releases, sign up at [trainindata](https://www.trainindata.com)
