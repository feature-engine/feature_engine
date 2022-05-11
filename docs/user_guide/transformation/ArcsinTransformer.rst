.. _arcsin:

.. currentmodule:: feature_engine.transformation

ArcsinTransformer
=================

The :class:`ArcsinTransformer()` applies the arcsin transformation to
numerical variables.

The :class:`ArcsinTransformer()` only works with numerical variables with values between 0 and +1. If the variable contains a value outside of this range, the transformer will raise an error.

Let's load the breast cancer dataset and  separate it into train and test sets (more details about the dataset :ref:`here <datasets>`).

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

    from feature_engine import transformation as vt
      
    #Load dataset
    breast_cancer = load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y = breast_cancer.target

  # Separate into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

Now we want to apply the arcsin transformation to variables in the dataframe:

.. code:: python

    vars_ = [
      'mean compactness',
      'mean concavity',
      'mean concave points',
      'mean fractal dimension',
      'smoothness error',
      'compactness error',
      'concavity error',
      'concave points error',
      'symmetry error',
      'fractal dimension error',
      'worst symmetry',
      'worst fractal dimension']

    # set up the variable transformer
    tf = vt.ArcsinTransformer(variables = vars_)

    # fit the transformer
    tf.fit(X_train)
    
    # test get_feature_names_out method
    feature_names = tf.get_feature_names_out()

The transformer does not learn any parameters. So we can go ahead and transform the
variables:

.. code:: python

    # transform the data
    train_t = tf.transform(X_train)

Finally, we can plot the original variable distribution:

.. code:: python

    # un-transformed variable
    X_train[vars_].hist(figsize=(20,20))

.. image:: ../../images/breast_cancer_raw.png

And now the distribution after the transformation:

.. code:: python

    # transformed variable
    train_t[vars_].hist(figsize=(20,20))


.. image:: ../../images/breast_cancer_arcsin.png

More details
^^^^^^^^^^^^

You can find more details about the :class:`ArcsinTransformer()` here:


- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/transformation/ReciprocalTransformer.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
