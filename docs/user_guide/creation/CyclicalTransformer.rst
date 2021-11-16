.. _cyclical_features:

.. currentmodule:: feature_engine.creation

CyclicalTransformer
===================

The :class:`CyclicalTransformer()` applies cyclical transformations to numerical
variables. The transformations return 2 new features per variable, according to:

- var_sin = sin(variable * (2. * pi / max_value))
- var_cos = cos(variable * (2. * pi / max_value))

where max_value is the maximum value in the variable, and pi is 3.14...

**Motivation**

There are some features that are cyclic by nature. For example the
hours of a day or the months in a year. In these cases, the higher values of
the variable are closer to the lower values. For example, December (12) is closer
to January (1) than to June (6). By applying a cyclical transformation we capture
this cycle or proximity between values.

**Examples**

In the code example below, we show how to obtain cyclical features from days and months
in a toy dataframe.

We first create a toy dataframe with the variables "days" and "months":

.. code:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.creation import CyclicalTransformer

    df = pd.DataFrame({
        'day': [6, 7, 5, 3, 1, 2, 4],
        'months': [3, 7, 9, 12, 4, 6, 12],
        })

Now we set up the transformer to find the maximum value automatically:

.. code:: python

    cyclical = CyclicalTransformer(variables=None, drop_original=True)

    X = cyclical.fit_transform(df)

The maximum values used for the transformation are stored in the attribute `max_values_`:

.. code:: python

    print(cyclical.max_values_)

.. code:: python

    {'day': 7, 'months': 12}

We can now see the new variables in the dataframe. Note that we set `drop_original=True`,
which means that the transformer will drop the original variables after the transformation.
If we had chosen False, the new variables will be added alongside the original ones.

.. code:: python

    print(X.head())

.. code:: python

          day_sin     day_cos  months_sin  months_cos
    1    -0.78183	  0.62349	      1.0	      0.0
    2         0.0	      1.0	     -0.5	 -0.86603
    3    -0.97493	-0.222521	     -1.0	     -0.0
    4     0.43388	-0.900969	      0.0	      1.0
    5     0.78183	  0.62349	  0.86603	     -0.5
    6     0.97493	-0.222521	      0.0	     -1.0
    7    -0.43388	-0.900969	      0.0	      1.0





