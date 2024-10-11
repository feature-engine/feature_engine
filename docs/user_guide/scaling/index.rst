.. -*- mode: rst -*-
.. _scaling_user_guide:

.. currentmodule:: feature_engine.scaling

Scaling
=======

`Feature scaling <https://www.blog.trainindata.com/feature-scaling-in-machine-learning/>`_
is the process of transforming the range of numerical features so that they fit within a
specific scale, usually to improve the performance and training stability of machine learning
models.

Scaling helps to normalize the input data, ensuring that each feature contributes proportionately
to the final result, particularly in algorithms that are sensitive to the range of the data,
such as gradient descent-based models (e.g., linear regression, logistic regression, neural networks)
and distance-based models (e.g., K-nearest neighbors, clustering).

Feature-engine's scalers replace the variables' values by the scaled ones. In this page, we
discuss the importance of scaling numerical features, and then introduce the various
scaling techniques supported by Feature-engine.

Importance of scaling
---------------------

Scaling is crucial in machine learning as it ensures that features contribute equally to model
training, preventing bias toward variables with larger ranges. Properly scaled data enhances the
performance of algorithms sensitive to the magnitude of input values, such as gradient descent
and distance-based methods. Additionally, scaling can improve convergence speed and overall model
accuracy, leading to more reliable predictions.


When apply scaling
------------------

- **Training:** Most machine learning algorithms require data to be scaled before training,
  especially linear models, neural networks, and distance-based models.

- **Feature Engineering:** Scaling can be essential for certain feature engineering techniques,
  like polynomial features.

- **Resampling:** Some oversampling methods like SMOTE and many of the undersampling methods
  clean data based on KNN algorithms, which are distance based models.


When Scaling Is Not Necessary
-----------------------------

Not all algorithms require scaling. For example, tree-based algorithms (like Decision Trees,
Random Forests, Gradient Boosting) are generally invariant to scaling because they split data
based on the order of values, not the magnitude.

Scalers
-------

.. toctree::
   :maxdepth: 1

   MeanNormalizationScaler
