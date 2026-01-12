.. _arcsinh_transformer:

.. currentmodule:: feature_engine.transformation

ArcSinhTransformer
==================

:class:`ArcSinhTransformer()` applies the inverse hyperbolic sine transformation
(arcsinh) to numerical variables. Also known as the pseudo-logarithm, this
transformation is useful for data that contains both positive and negative values.

The transformation is: x → arcsinh((x - loc) / scale)

Comparison to LogTransformer and ArcsinTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **LogTransformer**: `log(x)` requires `x > 0`. If your data contains zeros or negative values, you cannot use the standard LogTransformer directly. You would need to shift the data (e.g. `LogCpTransformer`) or remove non-positive values.
- **ArcsinTransformer**: `arcsin(sqrt(x))` is typically used for proportions/ratios bounded between 0 and 1. It is not suitable for general unbounded numerical data.
- **ArcSinhTransformer**: `arcsinh(x)` works for **all real numbers** (positive, negative, and zero). It handles zero gracefully (arcsinh(0) = 0) and is symmetric around zero.

When to use ArcSinhTransformer:
- Your data contains zeros or negative values (e.g., profit/loss, debt, temperature).
- You want a log-like transformation to stabilize variance or compress extreme values.
- You don't want to add an arbitrary constant (shift) to make values positive.

Intuitive Explanation of Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transformation includes optional `loc` (location) and `scale` parameters:

.. math::
    y = \text{arcsinh}\left(\frac{x - \text{loc}}{\text{scale}}\right)

- **Why scale?**
  The `arcsinh(x)` function is linear near zero (for small x) and logarithmic for large x.
  The "linear region" is roughly between -1 and 1.
  By adjusting the `scale`, you control which part of your data falls into this linear region versus the logarithmic region.
  - If `scale` is large, more of your data falls in the linear region (behavior close to original data).
  - If `scale` is small, more of your data falls in the logarithmic region (stronger compression of values).
  Common practice is to set `scale` to 1 or usage the standard deviation of the variable.

- **Why loc?**
  The `loc` parameter centers the data. The transition from negative logarithmic behavior to positive logarithmic behavior happens around `x = loc`.
  Common practice is to set `loc` to 0 or usage the mean of the variable.

References
~~~~~~~~~~

For more details on the inverse hyperbolic sine transformation:

1. `How should I transform non-negative data including zeros? <https://stats.stackexchange.com/questions/1444/how-should-i-transform-non-negative-data-including-zeros>`_ (StackExchange)
2. `Interpreting Treatment Effects: Inverse Hyperbolic Sine Outcome Variable <https://blogs.worldbank.org/en/impactevaluations/interpreting-treatment-effects-inverse-hyperbolic-sine-outcome-variable-and>`_ (World Bank Blog)
3. `Burbidge, J. B., Magee, L., & Robb, A. L. (1988). Alternative transformations to handle extreme values of the dependent variable. Journal of the American Statistical Association. <https://www.jstor.org/stable/2288929>`_

Example
~~~~~~~

Let's create a dataframe with positive and negative values and apply the arcsinh
transformation:

Unlike :class:`LogTransformer()`, :class:`ArcSinhTransformer()` can handle
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

:class:`ArcSinhTransformer()` supports location and scale parameters to
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

:class:`ArcSinhTransformer()` supports inverse transformation to recover
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
