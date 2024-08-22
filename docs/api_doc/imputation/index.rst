

.. -*- mode: rst -*-

.. currentmodule:: feature_engine.imputation

Missing Data Imputation
=======================

Missing data refers to the absence of observed values in a dataset and is a common occurrence in any real-world data
science project. In data science, missing data can lead to biased analysis, inaccurate predictions, and reduced reliability
of models. Therefore, handling missing data has become one of the most important steps in a data preprocessing pipeline.

Feature-engine supports several imputation techniques to handle missing data. Here, we provide an overview of each of
the supported methods.

.. toctree::
   :maxdepth: 1
   :hidden:

   MeanMedianImputer
   ArbitraryNumberImputer
   EndTailImputer
   CategoricalImputer
   RandomSampleImputer
   AddMissingIndicator
   DropMissingData


Missing data mechanisms
-----------------------

Data can go missing for several reasons, including:

- In surveys, respondents may choose not to answer specific questions due to privacy concerns or simply overlooking them.
- In healthcare data, not every patient might undergo a study on the efficacy of new medications due to logistical or financial constraints.
- Errors in data collection and storage can also lead to missing values.

The mechanisms that introduce missing data are known as completely at random (MCAR), missing at random (MAR) or missing
not at random (NMAR).


Consequences of missing data
----------------------------

Missing data can significantly impact machine learning models and statistical analysis for several reasons:

- It introduces bias into machine learning model predictions or statistical tests.
- Certain machine learning models, such as those found in Scikit-learn, cannot handle datasets with missing values.

Popular machine learning algorithms like linear regression, logistic regression, support vector machine (SVM), or k-nearest
neighbors (kNN) are not equipped to manage datasets containing NaN (Not a Number) or null values. Consequently, attempting
to fit these models with such incomplete data will result in errors.

These factors underscore the importance of addressing missing data prior to model training, highlighting the necessity of
employing data imputation techniques.

Missing Data Imputation
-----------------------

Missing data imputation refers to the process of estimating and replacing missing values within a dataset. It involves
filling in the missing values with estimated values based on the known information in the dataset.

There are two types of missing data imputation: univariate and multivariate imputation.

Univariate data imputation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Univariate imputation addresses missing data within a variable solely based on information within that variable, without
considering other variables in the dataset.

For instance, consider a dataframe with exam results of 50 college students, and 5 data points are missing. Univariate
imputation fills these 5 missing values based on operations such as mean, median, or mode of the 45 observed values.
Alternatively, the missing data can be filled with arbitrary predefined values, such as -1, 0, 999, -999, or 'Missing', among others.

Multivariate data imputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In multivariate data imputation, we utilize observations from other variables in the dataset to estimate the values of
missing observations. This method essentially imputes missing values by treating the imputation as a regression, using
algorithms such as k-nearest neighbors or linear regression to estimate the missing values.

For example, let's say we have a dataset containing information on students' grades, ages, and IQ scores, all of which
have missing values. In this scenario, we can predict the missing grade values by employing a regression model trained
on existing grade data, using age and IQ as predictors. Subsequently, we can apply the same regression imputation approach
to the other variables (age and IQ) in subsequent iterations.

Feature-engine currenty supports univariate imputation strategies. For multivariate imputation, check out Scikit-learn's `iterative imputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>`_.


Feature-engine's imputation methods
-----------------------------------

Feature-engine supports the following data imputation methods

- Mean-median imputation
- Arbitrary number imputation
- End tail imputation
- Random sample imputation
- Frequent category imputation
- Categorical imputation
- Complete case analysis
- Adding missing indicators

|

Feature-engine's imputers main characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

================================== ===================== ======================= ====================================================================================
    Transformer                     Numerical variables	  Categorical variables	    Description
================================== ===================== ======================= ====================================================================================
:class:`MeanMedianImputer()`	        √	                 ×	                    Replaces missing values by the mean or median
:class:`ArbitraryNumberImputer()`	    √	                 x	                    Replaces missing values by an arbitrary value
:class:`EndTailImputer()`	            √	                 ×	                    Replaces missing values by a value at the end of the distribution
:class:`CategoricalImputer()`           √	                 √	                    Replaces missing values by the most frequent category or by an arbitrary value
:class:`RandomSampleImputer()`	        √	                 √	                    Replaces missing values by random value extractions from the variable
:class:`AddMissingIndicator()`	        √	                 √	                    Adds a binary variable to flag missing observations
:class:`DropMissingData()`	            √	                 √	                    Removes observations with missing data from the dataset
================================== ===================== ======================= ====================================================================================


Mean-Median Imputation
~~~~~~~~~~~~~~~~~~~~~~

Mean-median imputation replaces missing values in a numerical variable with the median or mean value of that variable.

If a variable follows a normal distribution, both the mean and the median are suitable options since they are equivalent.
However, if a variable is skewed, median imputation is preferable as mean imputation can introduce bias toward the tail of the distribution.

This imputation method is suited if the data is missing completely at random (MCAR). If data is MCAR, then it is fair to
assume that the missing values are close in value to the majority, that is, to the mean or median of the distribution.

**Advantages**:

- Fast and easy method to obtain complete data.

**Limitations**:

- Distorts the variance within a variable, as well as the covariance and correlation with other variables in the dataset.

Data imputed with the mean or median is commonly used to train linear regression and logistic regression models.

The :class:`MeanMedianImputer()` implements mean-median imputation.

Arbitrary Number Imputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Arbitrary number imputation replaces missing values in a numerical variable with an arbitrary number. Common values used
for replacements are 0, 999, -999 (or other 9 combinations), or -1 (if the distribution is positive).

This imputation method is perfectly suited if the data is missing not at random (MNAR). This is because the method will
flag the missing values with a predefined arbitrary value instead of replacing them with statistical estimates that make
nan values look like the majority of the observations.

**Advantages**:

- Fast and easy way to obtain complete data.
- Flags missing values.

**Limitations**:

- Distorts in the variance within a variable, as well as the covariance and correlation with other variables in the dataset.
- It might hide or create outliers.
- Need to be careful not to choose an arbitrary value that is too similar to the mean or median.

Some models can be effectively trained with data that has undergone arbitrary number imputation, such as tree-based models,
kNN, SVM, and ensemble models.

The :class:`ArbitraryNumberImputer()` implements arbitrary number imputation.

End Tail Imputation
~~~~~~~~~~~~~~~~~~~

End tail imputation replaces missing values in a numerical variable with an arbitrary number located at the tail of the
variable's distribution.

We can select the imputation value in one of 2 ways depending on the variable's distribution:

- If it’s a normal distribution, the value can be set at the mean plus or minus 3 times the standard deviation.
- If it’s a skewed distribution, the value can be set using the IQR.

This method is suitable for MNAR data. This is because this method will flag the missing value instead of replacing it
with a value that is similar to the majority of observations.

**Advantages**:

- Fast and easy way to obtain complete datasets.
- Automates arbitrary value imputation.
- Flags missing values.

**Limitations**:

- Distortion of the original variance within a variable, as well as the covariance and correlation with other variables in the dataset.
- It might hide outliers.

Models like tree-based models, tree based models can be effectively trained on data imputed with end tail imputation.

The :class:`EndTailImputer()` implements end tail imputation.

Random Sample Imputation
~~~~~~~~~~~~~~~~~~~~~~~~

Random sample imputation replaces missing values in both numerical and categorical variables with a random value drawn
from a distribution of that variable.

Since the replacement is drawn from the distribution of the original variable, the variance of the imputed data will be
preserved. However, due to its randomness, we could obtain different imputation values on different code executions, which
would lead to different machine learning model predictions. Therefore, make sure to set a proper seed during the imputation.

Random sample imputation is useful when we don't want to distort the distribution of the variable.

**Advantages**:

- Preserves the variance of a variable.

**Limitations**:

- Randomness.
- Distorts the relation with other variables.
- This imputation model is computationally more expensive than other methods.

The :class:`RandomSampleImputer()` implements random sample imputation.

Frequent Category imputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Frequent category imputation replaces missing values in categorical variables with the most frequent category in that variable.

Although :class:`CategoricalImputer()` can impute both numerical and categorical variables, in practice frequent category
imputation is more commonly used for categorical variable imputation.

This method is suited if the data is MCAR, as this imputation method replaces missing values with the most common
observation in our variable.

**Advantages**:

- Fast and easy method to obtain complete data.

**Limitations**:

- Imputed values can distort the correlation with other variables.
- It can lead to an over-representation of the most frequent category.

Therefore, it’s best to use this method if the missing values constitute a small percentage of the observations.

Tree-based models, kNN, SVM, and ensemble models can be effectively trained on data imputed with frequent category imputation.

The :class:`CategoricalImputer()` implements frequent category imputation.

Categorical imputation
~~~~~~~~~~~~~~~~~~~~~~

During categorical imputation, we replace missing values in a categorical variable with a specific new label, such as
‘Missing’ or 'NaN' for example. In essence, it consists of treating the missing observations as a category in itself.

This method is suited for MNAR data because it marks the missing values with a new label, instead of replacing them with
statistical estimates that may introduce bias in our data.

**Advantages**:

- Fast and easy way to obtain complete data.
- Flags missing values.
- No assumption made on the data.

**Limitations**:

- If the proportion of missing values is little, creating an additional category might introduce noise.

The :class:`CategoricalImputer()` implements categorical imputation.

Adding Missing Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~

Adding missing indicators consists in adding binary variables to highlight if the values are missing. The missing
indicator takes the value 0 if there is an observed value and 1 if the value was missing.

Adding missing indicators does not replace the missing data in itself. They just add the information to the data that
some values were missing. Therefore, this method is never used alone. Normally, it’s accompanied with other imputation
methods, such as mean-median for numerical data or frequent category imputation for categorical data.

**Advantages**:

- Captures the importance of missing values.

**Limitations**:

- Expands the dimensionality of the data.

The :class:`AddMissingIndicator()` adds missing indicators to the dataset.


Complete case analysis
~~~~~~~~~~~~~~~~~~~~~~

Dropping missing data is the simplest method to deal with missing data. This procedure is known as complete case analysis
or listwise deletion, meaning that the entire row will be excluded from analysis if any single value is missing.

This method is best suited for MCAR data and if the proportion of missing values is relatively small.

**Advantages**:

- Fast and easy way to obtain complete data.

**Limitations**:

- Reducing the sample size of available data.
- Potentially creating bias in our data, hence affecting data analysis.

The :class:`DropMissingData()` implements complete case analysis.

Wrapping up
-----------

All Feature-engine supported data imputation methods are single imputation methods, or better said, univariate imputation
methods.

There are alternative data imputation techniques that data scientists could also use, like:

**Multiple Imputation**: Multiple imputation generates several imputed datasets by randomly imputing missing values in
each of the dataset. Suitable if the data is MAR.

**Cold Deck Imputation**: Cold deck imputation replaces missing values with values borrowed from a historical dataset.

**Hot Deck Imputation**: Hot deck imputation selects imputed values from a similar subset of observed data within the same
dataset.

**Multiple imputation of chained equations (MICE)**: MICE is a way of estimating the missing data as a regression based
on the other variables in the dataset. It uses multiple rounds of imputation to improve the estimates at each iteration.

Additional resources
--------------------

For tutorials about missing data imputation methods check out these resources:


.. figure::  ../../images/feml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-machine-learning

   Feature Engineering for Machine Learning

.. figure::  ../../images/fetsf.png
   :width: 300
   :figclass: align-center
   :align: right
   :target: https://www.trainindata.com/p/feature-engineering-for-forecasting

   Feature Engineering for Time Series Forecasting

|
|
|
|
|
|
|
|
|
|

Our book:

.. figure::  ../../images/cookbook.png
   :width: 200
   :figclass: align-center
   :align: left
   :target: https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587

   Python Feature Engineering Cookbook

|
|
|
|
|
|
|
|
|
|
|
|
|

Both our book and courses are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.
