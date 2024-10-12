.. _mean_normalization_scaler:

.. currentmodule:: feature_engine.scaling

MeanNormalizationScaler
=======================

:class:`MeanNormalizationScaler()` scales variables using mean normalization. With mean normalization,
we center the distribution around 0, and rescale the distribution to the variable's value range,
so that its values vary between -1 and 1. This is accomplished by subtracting the mean of the feature
and then dividing by its range (i.e., the difference between the maximum and minimum values).

The :class:`MeanNormalizationScaler()` only works with non-constant numerical variables.
If the variable is constant, the scaler will raise an error.

Python example
--------------

We'll show how to use :class:`MeanNormalizationScaler()` through a toy dataset. Let's create
a toy dataset:

.. code:: python

    import pandas as pd
    from feature_engine.scaling import MeanNormalizationScaler

    df = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Height": [1.80, 1.77, 1.90, 2.00],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
        })

    print(df)

The dataset looks like this:

.. code:: python

        Name        City  Age  Height  Marks                 dob
    0    tom      London   20    1.80    0.9 2020-02-24 00:00:00
    1   nick  Manchester   21    1.77    0.8 2020-02-24 00:01:00
    2  krish   Liverpool   19    1.90    0.7 2020-02-24 00:02:00
    3   jack     Bristol   18    2.00    0.6 2020-02-24 00:03:00

We see that the only numerical features in this dataset are **Age**, **Marks**, and **Height**. We want
to scale them using mean normalization.

First, let's make a list with the variable names:

.. code:: python

    vars = [
      'Age',
      'Marks',
      'Height',
    ]

Now, let's set up :class:`MeanNormalizationScaler()`:

.. code:: python

    # set up the scaler
    scaler = MeanNormalizationScaler(variables = vars)

    # fit the scaler
    scaler.fit(df)
    
The scaler learns the mean of every column in *vars* and their respective range.
Note that we can access these values in the following way:

.. code:: python

    # access the parameters learned by the scaler
    print(f'Means: {scaler.mean_}')
    print(f'Ranges: {scaler.range_}')

We see the features' mean and value ranges in the following output:

.. code:: python

    Means: {'Age': 19.5, 'Marks': 0.7500000000000001, 'Height': 1.8675000000000002}
    Ranges: {'Age': 3.0, 'Marks': 0.30000000000000004, 'Height': 0.22999999999999998}

We can now go ahead and scale the variables:

.. code:: python

    # scale the data
    df = scaler.transform(df)
    print(df)

In the following output, we can see the scaled variables:

.. code:: python

        Name        City       Age    Height     Marks                 dob
    0    tom      London  0.166667 -0.293478  0.500000 2020-02-24 00:00:00
    1   nick  Manchester  0.500000 -0.423913  0.166667 2020-02-24 00:01:00
    2  krish   Liverpool -0.166667  0.141304 -0.166667 2020-02-24 00:02:00
    3   jack     Bristol -0.500000  0.576087 -0.500000 2020-02-24 00:03:00

We can restore the data to itsoriginal values using the inverse transformation:

.. code:: python

    # inverse transform the dataframe
    df = scaler.inverse_transform(df)
    print(df)

In the following data, we see the scaled variables returned to their oridinal representation:

.. code:: python

        Name        City  Age  Height  Marks                 dob
    0    tom      London   20    1.80    0.9 2020-02-24 00:00:00
    1   nick  Manchester   21    1.77    0.8 2020-02-24 00:01:00
    2  krish   Liverpool   19    1.90    0.7 2020-02-24 00:02:00
    3   jack     Bristol   18    2.00    0.6 2020-02-24 00:03:00


Additional resources
--------------------

For more details about this and other feature engineering methods check out
these resources:


.. figure::  ../../images/feml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-machine-learning

   Feature Engineering for Machine Learning

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

Or read our book:

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

Both our book and course are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.