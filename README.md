# Feature Engine

Feature-engine is a Python library that contains several transformers to engineer features for use in machine learning models. Feature-engine's transformers follow Scikit-learn like functionality with fit() and transform() methods to first learn the transforming paramenters from data and then transform the data.
Current Feature-engine's transformers include functionality for:

* Missing data imputation
* Categorical variable encoding
* Outlier removal
* Discretisation
* Numerical Variable Transformation

## Important Links

* Documentation: http://feature-engine.readthedocs.io
* Home page: https://www.trainindata.com/feature-engine

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

See more usage examples in the jupyter notebooks in the example folder of this repository, or in the documentation: http://feature-engine.readthedocs.io

### License

BSD 3-Clause

### Authors

* **Soledad Galli** - *Initial work* - [Feature Engineering Online Course](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=FEATENGREPO).


### References

Many of the engineering and encoding functionality is inspired by this [series of articles from the 2009 KDD competition](http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf).

To learn more about the rationale, functionality, pros and cos of each imputer, encoder and transformer, refer to the [Feature Engineering Online Course](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=FEATENGREPO)

For a summary of the methods check this [presentation](https://speakerdeck.com/solegalli/engineering-and-selecting-features-for-machine-learning) and this [article](https://www.trainindata.com/post/feature-engineering-comprehensive-overview)

To stay alert of latest releases, sign up at [trainindata](https://www.trainindata.com)