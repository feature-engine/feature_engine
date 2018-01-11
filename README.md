# Feature Engine

Feature Engine is a python library that contains several transformers to engineer features for use in machine learning models.
The transformers follow scikit-learn like functionality. They first learn the imputing or encoding methods from the training sets, and subsequently transform the dataset.
Currently there trasformers include:

* Missing value imputation
* Categorical variable encoding
* Outlier removal

## Important Links

Documentation: http://feature-engine.readthedocs.io

### Imputing Methods

* MeanMedianImputer
* RandomSampleImputer
* EndTailImputer
* na_capturer
* CategoricalImputer
* ArbitraryImputer

### Encoding Methods
* CategoricalEncoder
	* count: number of observations per label 
	* frequency : percentage of observation per label
	* ordinal : labels are labelled according to increasing target mean value
	* mean : target mean per label
	* ratio : target probability ratio per label
	* woe : weight of evidence
* RareLabelEncoder

### Outlier Handler
* Windsorizer

### Installing

```
git clone https://gitlab.com/datascientistcoach/feature_engine.git
```

### Usage

```
from categorical_encoder import CategoricalEncoder, RareLabelEncoder

rare_encoder = RareLabelEncoder(tol = 0.05, n_categories=5)
rare_encoder.fit(data, variables = ['Cabin', 'Age'])
data_encoded = rare_encoder.transform(data)
```

See more examples in the example section

### Examples

You can find jupyter notebooks in the examples directory, with directions on how to use this package.

### License

BSD 3-Clause

### Authors

* **Soledad Galli** - *Initial work* - [Feature Engineering Online Course](https://www.udemy.com/feature-engineering-for-machine-learning)


### References

* Forthcoming