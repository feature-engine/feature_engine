.. _arcsin:

.. currentmodule:: feature_engine.scaling

MeanNormalizationScaler
======================

The :class:`MeanNormalizationScaler()` scales one or more columns using
the mean normalization scaling technique.

Mean normalization scales each feature in the dataset by subtracting the mean of that feature
and then dividing by the range (i.e., the difference between the maximum and minimum values) of
that feature. The resulting feature values are centered around zero, but they are not standardized
to have a unit variance, nor are they normalized to a fixed range.

The :class:`MeanNormalizationScaler()` only works with non-constant numerical variables.
If the variable is constant, the scaler will raise an error.

Example
~~~~~~~

Let's dive into the mean normalization scaler. First, we create a toy dataset:

.. code:: python

    import numpy as np
    import pandas as pd
    from feature_engine.scaling import MeanNormalizationScaler

    df = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Height": [1.80, 1.77, 1.90, 2.00],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
        })

    print(df)

The dataset looks like this

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

    vars_ = [
      'Age',
      'Marks',
      'Height',
    ]

Now, let's set up the mean normalization scaler :

.. code:: python

    # set up the scaler
    scaler = MeanNormalizationScaler(variables = vars_)

    # fit the scaler
    scaler.fit(df)
    
The scaler learns the mean of every column in *vars_* and the respective range.
Note that we can access these values in the following way:

.. code:: python

    # access the parameters learned by the scaler
    means = scaler.mean_
    ranges = scaler.range_

    print(f'Means: {means}')
    print(f'Ranges: {ranges}')

This is the result:

.. code:: python

    Means: {'Age': 19.5, 'Marks': 0.7500000000000001, 'Height': 1.8675000000000002}
    Ranges: {'Age': 3.0, 'Marks': 0.30000000000000004, 'Height': 0.22999999999999998}

Note that we can access to these parameters only once the scaler has benn fit.

We can now go ahead and scale the variables:

.. code:: python

    # scale the data
    df = scaler.transform(df)
    print(df)

And that's it, now the selected variables have been scaled using mean normalization,
as we can see:

.. code:: python

        Name        City       Age    Height     Marks                 dob
    0    tom      London  0.166667 -0.293478  0.500000 2020-02-24 00:00:00
    1   nick  Manchester  0.500000 -0.423913  0.166667 2020-02-24 00:01:00
    2  krish   Liverpool -0.166667  0.141304 -0.166667 2020-02-24 00:02:00
    3   jack     Bristol -0.500000  0.576087 -0.500000 2020-02-24 00:03:00

We can at everytime returning to the original values using the inverse transformation.

.. code:: python

    # inverse transform the dataframe
    df = scaler.inverse_transform(df)
    print(df)

As promised

.. code:: python

        Name        City  Age  Height  Marks                 dob
    0    tom      London   20    1.80    0.9 2020-02-24 00:00:00
    1   nick  Manchester   21    1.77    0.8 2020-02-24 00:01:00
    2  krish   Liverpool   19    1.90    0.7 2020-02-24 00:02:00
    3   jack     Bristol   18    2.00    0.6 2020-02-24 00:03:00

