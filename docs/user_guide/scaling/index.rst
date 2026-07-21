.. -*- mode: rst -*-
.. _scaling_user_guide:

.. currentmodule:: feature_engine.scaling

Feature Scaling
===============

`Feature scaling <https://www.blog.trainindata.com/feature-scaling-in-machine-learning/>`_
is the process of transforming the range of numerical features so that they fit within a
specific scale, usually to improve the performance and training stability of machine learning
models.

Scaling helps to normalise the input data, ensuring that each feature contributes proportionately
to the final result, particularly in algorithms that are sensitive to the range of the data,
such as gradient descent-based models (e.g., linear regression, logistic regression, neural networks)
and distance-based models (e.g., K-nearest neighbours, clustering).

Feature-engine's scalers replace the variables' values by the scaled ones. In this page, we
discuss the importance of scaling numerical features, and then introduce the various
scaling techniques supported by feature-engine.

Importance of scaling
---------------------

Scaling is crucial in machine learning as it ensures that features contribute equally to model
training, preventing bias toward variables with larger ranges. Properly scaled data enhances the
performance of algorithms sensitive to the magnitude of input values, such as gradient descent
and distance-based methods. Additionally, scaling can improve convergence speed and overall model
accuracy, leading to more reliable predictions.


When to apply scaling
---------------------

- **Training:** Most machine learning algorithms require data to be scaled before training,
  especially linear models, neural networks, and distance-based models.

- **Feature engineering:** Scaling can be essential for certain feature engineering techniques,
  like polynomial features.

- **Resampling:** Some oversampling methods like SMOTE and many of the undersampling methods
  resample data based on KNN algorithms, which are distance based models.

- **Dimensionality reduction:** Principal component analysis (PCA) and other dimensionality reduction
  methods are distance based, and as such, sensitive to the scale of the features (more details in our
  course `Clustering and Dimensionality Reduction <https://www.trainindata.com/p/clustering-and-dimensionality-reduction>`_.)

When scaling is not necessary
-----------------------------

Not all algorithms require scaling. For example, tree-based algorithms (like Decision Trees,
Random Forests, Gradient Boosting) are generally invariant to scaling because they split data
based on the order of values, not their magnitude.

Scalers
-------

.. toctree::
   :maxdepth: 1

   MeanNormalizationScaler
   GroupStandardScaler
