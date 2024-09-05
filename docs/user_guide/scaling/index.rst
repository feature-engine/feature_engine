.. -*- mode: rst -*-
.. _encoding_user_guide:

.. currentmodule:: feature_engine.scaling

Scaling
=======

Scaling in data science is the process of transforming the range of your numerical features
so that they fit within a specific scale, usually to improve the performance and training
stability of machine learning models. Scaling helps to normalize the input data, ensuring
that each feature contributes proportionately to the final result, particularly in
algorithms that are sensitive to the range of the data, such as gradient descent-based models
(e.g., linear regression, logistic regression, neural networks) and distance-based
models (e.g., K-nearest neighbors, clustering).

Feature-engine's scalers replace the variables' values by the scaled ones.
In this page, we will discuss the importance of scaling numerical features,
and then introduce the various scaling techniques supported by Feature-engine.

Numerical features
--------------------

Numerical variables are those whose values have an order relation between them.
Their values are in general numbers.

*Price* is a categorical feature that can take values such as *100*, *50* or
*45.50*, and, given two prices, it is always possible to define which one is the biggest.
Similarly, *height* is another categorical feature, as well as *distance*, or *length*.


Identifying numerical features
--------------------------------

We can identify numerical features by asking ourself the question: given two values of this feature.
Is it always possible to order them?their data types.


Importance of scaling
---------------------

Most machine learning algorithms, like *linear regression*, *support vector machines* 
and *logistic regression*, require input data to be numeric because they use numerical 
computations to learn the relationship between the predictor features and the target 
variable. These algorithms are not inherently capable of interpreting categorical data. 
Thus, categorical encoding is a crucial step that ensures that the input data is compatible 
with the expectations of the machine learning models.

Some implementations of *decision tree* based algorithms can directly handle categorical data. 
We'd still recommend encoding categorical features, for example, to reduce cardinality and 
account for unseen categories.


When apply scaling
------------------

- **Training:** Most machine learning algorithms require data to be scaled before training,
  especially linear models, neural networks, and distance-based models.

- **Feature Engineering:** Scaling can be essential for certain feature engineering techniques,
  like polynomial features.

- **Pipelines:** In practice, scaling is often applied within a preprocessing pipeline to
  ensure it's consistently applied to both training and test data.


When Scaling Is Not Necessary
-----------------------------
Not all algorithms require scaling. For example, tree-based algorithms (like Decision Trees,
Random Forests, Gradient Boosting) are generally invariant to scaling because they split data
based on the order of values, not the magnitude.


Scaling methods
----------------

There are various methods to transform categorical variables into numerical features. One hot 
encoding and ordinal encoding are the most commonly used, but other methods can mitigate high 
cardinality and account for unseen categories.

In the rest of this page, we'll introduce various methods for encoding categorical data, and 
highlight the Feature-engine transformer that can carry out this transformation.

Mean Normalization Scaler
~~~~~~~~~~~~~~~~~~~~~~~~~

Mean normalization scales each feature in the dataset by subtracting the mean of that feature
and then dividing by the range (i.e., the difference between the maximum and minimum values) of
that feature. The resulting feature values are centered around zero, but they are not standardized
to have a unit variance, nor are they normalized to a fixed range.

Feature-engine's :class:`MeanNormalizationScaler` implements mean normalization scaling.



Scalers
-------

.. toctree::
   :maxdepth: 1

   MeanNormalizationScaler
