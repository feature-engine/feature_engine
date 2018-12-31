# Feature Engine

Feature Engine is a python library that contains several transformers to engineer features for use in machine learning models.
The transformers follow scikit-learn like functionality. They first learn the imputing or encoding methods from the training set, and subsequently transform the dataset.
Currently the trasformers include functionality for:

* Missing value imputation
* Categorical variable encoding
* Outlier removal
* Discretisation
* Numerical Variable Transformation

## Important Links

Documentation: http://feature-engine.readthedocs.io

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
* Windsorizer
* ArbitraryOutlierCapper

### Discretisation methods
* EqualFrequencyDiscretiser
* EqualWidthDiscretiser
* DecisionTreeDiscretiser

### Variable Transformation methods
* LogTransformer
* ReciprocalTransformer
* ExponentialTransformer
* BoxCoxTransformer

### Installing

```
pip install feature_engine
```
or

```
git clone https://gitlab.com/datascientistcoach/feature_engine.git
```

### Usage

```
from feature_engine.categorical_encoders import RareLabelEncoder

rare_encoder = RareLabelEncoder(tol = 0.05, n_categories=5)
rare_encoder.fit(data, variables = ['Cabin', 'Age'])
data_encoded = rare_encoder.transform(data)
```

See more usage examples in the jupyter notebooks in the example section

### Examples

You can find jupyter notebooks in the examples folder, with directions on how to use this package and its multiple transformers.

### License

BSD 3-Clause

### Authors

* **Soledad Galli** - *Initial work* - [Feature Engineering Online Course](https://www.udemy.com/feature-engineering-for-machine-learning)


### References

Most of the engineering and encoding functionality is inspired by this [series of articles from the 2009 KDD competition](http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf)

To learn more about the rationale, functionality, pros and cos of each imputer, encoder and transformer, refer to the Feature Engineering Online Course](https://www.udemy.com/feature-engineering-for-machine-learning)
 