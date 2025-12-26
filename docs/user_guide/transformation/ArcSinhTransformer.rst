.. _arcsinh_transformer:

.. currentmodule:: feature_engine.transformation

ArcSinhTransformer
==================

The :class:`ArcSinhTransformer()` applies the inverse hyperbolic sine transformation
(arcsinh) to numerical variables. Also known as the pseudo-logarithm, this
transformation is useful for data that contains both positive and negative values.

The transformation is: x → arcsinh((x - loc) / scale)

For large |x|, arcsinh(x) behaves like ln(|x|) + ln(2), providing similar
variance-stabilizing properties as the log transformation. For small |x|,
it behaves approximately linearly (x → x). This makes it ideal for variables
like net worth, profit/loss, or any metric that can be positive or negative.

Unlike the :class:`LogTransformer()`, the :class:`ArcSinhTransformer()` can handle
zero and negative values without requiring any preprocessing.

Example
~~~~~~~

Let's create a dataframe with positive and negative values and apply the arcsinh
transformation:

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine.transformation import ArcSinhTransformer

    # Create sample data with positive and negative values
    np.random.seed(42)
    X = pd.DataFrame({
        'profit': np.random.randn(1000) * 10000,  # Values from -30000 to 30000
        'net_worth': np.random.randn(1000) * 50000,
    })

    # Separate into train and test
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)

Now let's set up the ArcSinhTransformer:

.. code:: python

    # Set up the arcsinh transformer
    tf = ArcSinhTransformer(variables=['profit', 'net_worth'])

    # Fit the transformer
    tf.fit(X_train)

The transformer does not learn any parameters when applying the fit method. It does
check however that the variables are numerical.

We can now transform the variables:

.. code:: python

    # Transform the data
    train_t = tf.transform(X_train)
    test_t = tf.transform(X_test)

The arcsinh transformation compresses extreme values while preserving the sign:

.. code:: python

    # Compare original and transformed distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    X_train['profit'].hist(ax=axes[0], bins=50)
    axes[0].set_title('Original profit')

    train_t['profit'].hist(ax=axes[1], bins=50)
    axes[1].set_title('Transformed profit')

    plt.tight_layout()

Using loc and scale parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`ArcSinhTransformer()` supports location and scale parameters to
center and normalize data before transformation:

.. code:: python

    # Center around mean and scale by std
    tf = ArcSinhTransformer(
        variables=['profit'],
        loc=X_train['profit'].mean(),
        scale=X_train['profit'].std()
    )

    tf.fit(X_train)
    train_t = tf.transform(X_train)

Inverse transformation
~~~~~~~~~~~~~~~~~~~~~~

The :class:`ArcSinhTransformer()` supports inverse transformation to recover
the original values:

.. code:: python

    # Transform and then inverse transform
    train_t = tf.transform(X_train)
    train_recovered = tf.inverse_transform(train_t)

    # Values should match original
    np.allclose(X_train['profit'], train_recovered['profit'])

API Reference
-------------

.. autoclass:: ArcSinhTransformer
    :members:
    :inherited-members:
