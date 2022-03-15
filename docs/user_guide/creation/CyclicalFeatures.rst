.. _cyclical_features:

.. currentmodule:: feature_engine.creation

CyclicalFeatures
================

The :class:`CyclicalFeatures()` applies cyclical transformations to numerical
variables. The transformations return 2 new features per variable, according to:

- var_sin = sin(variable * (2. * pi / max_value))
- var_cos = cos(variable * (2. * pi / max_value))

where max_value is the maximum value in the variable, and pi is 3.14...

**Motivation**

There are some features that are cyclic by nature. For example the
hours of a day or the months in a year. In these cases, the higher values of
the variable are closer to the lower values. For example, December (12) is closer
to January (1) than to June (6). By applying a cyclical transformations we capture
this cycle or proximity between values.

**Examples**

In the code example below, we show how to obtain cyclical features from days and months
in a toy dataframe.

We first create a toy dataframe with the variables "days" and "months":

.. code:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.creation import CyclicalFeatures

    df = pd.DataFrame({
        'day': [6, 7, 5, 3, 1, 2, 4],
        'months': [3, 7, 9, 12, 4, 6, 12],
        })

Now we set up the transformer to find the maximum value automatically:

.. code:: python

    cyclical = CyclicalFeatures(variables=None, drop_original=True)

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

            day_sin   day_cos    months_sin    months_cos
    0 -7.818315e-01  0.623490  1.000000e+00  6.123234e-17
    1 -2.449294e-16  1.000000 -5.000000e-01 -8.660254e-01
    2 -9.749279e-01 -0.222521 -1.000000e+00 -1.836970e-16
    3  4.338837e-01 -0.900969 -2.449294e-16  1.000000e+00
    4  7.818315e-01  0.623490  8.660254e-01 -5.000000e-01


Motivation
----------

Let's discuss more the logic behind using the sine and cosine to transform cyclical
or periodic variables like months of the year, or days of the week.

We mentioned that with cyclical or periodic features, values that are very different in
absolute magnitude are actually close. For example, January is close to December, even
though their absolute magnitude suggests otherwise.

We can use periodic functions like sine and cosine, to transform cyclical features and
help machine learning models pick up their intrinsic nature.

Let's create a toy dataframe and explain this in more detail:

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame([i for i in range(24)], columns=['hour'])

Our dataframe looks like this:

.. code:: python

    df.head()

       hour
    0     0
    1     1
    2     2
    3     3
    4     4

Let's now create the sine and cosine features to understand more their nature:

.. code:: python

    cyclical = CyclicalFeatures(variables=None)

    df = cyclical.fit_transform(df)

    print(df.head())

.. code:: python

       hour  hour_sin  hour_cos
    0     0  0.000000  1.000000
    1     1  0.269797  0.962917
    2     2  0.519584  0.854419
    3     3  0.730836  0.682553
    4     4  0.887885  0.460065


Let's now plot the hour variable against its sine transformation. We add perpendicular
lines to flag the hours 0 and 22.

.. code:: python

    plt.scatter(df["hour"], df["hour_sin"])

    # Axis labels
    plt.ylabel('Sine of hour')
    plt.xlabel('Hour')
    plt.title('Sine transformation')

    plt.vlines(x=0, ymin=-1, ymax=0, color='g', linestyles='dashed')
    plt.vlines(x=22, ymin=-1, ymax=-0.25, color='g', linestyles='dashed')

After the transformation, we see that 0 and 22 are actually closer to each other (follow
the dashed lines).

.. image:: ../../images/hour_sin.png

The problem with trigonometric transformations, is that, because they are periodic,
2 different observations can also return similar values after the transformation. let's
explore that:

.. code:: python

    plt.scatter(df["hour"], df["hour_sin"])

    # Axis labels
    plt.ylabel('Sine of hour')
    plt.xlabel('Hour')
    plt.title('Sine transformation')

    plt.hlines(y=0, xmin=0, xmax=11.5, color='r', linestyles='dashed')

    plt.vlines(x=0, ymin=-1, ymax=0, color='g', linestyles='dashed')
    plt.vlines(x=11.5, ymin=-1, ymax=0, color='g', linestyles='dashed')

In the plot below, we see that the hours 0 and 11.5 obtain very similar values after the
sine transformation. So how can we differentiate them?

.. image:: ../../images/hour_sin2.png

We need to use the 2 transformations together, sine and cosine, to fully code the
information of the hour. Adding the cosine function, which is out-of-phase with the sine
function, breaks the symmetry and gives each hour a unique codification. Let's explore
that:

.. code:: python

    plt.scatter(df["hour"], df["hour_sin"])
    plt.scatter(df["hour"], df["hour_cos"])

    # Axis labels
    plt.ylabel('Sine and cosine of hour')
    plt.xlabel('Hour')
    plt.title('Sine and Cosine transformation')


    plt.hlines(y=0, xmin=0, xmax=11.5, color='r', linestyles='dashed')

    plt.vlines(x=0, ymin=-1, ymax=1, color='g', linestyles='dashed')
    plt.vlines(x=11.5, ymin=-1, ymax=1, color='g', linestyles='dashed')

The hour 0, after the transformation, takes the values of sine 0 and cosine 1, which
makes it different from the hour 11.5, which takes values of sine 0 and cosine -1. In
other words, with the 2 functions together, we are able to distinguish all observations
within our variable.

.. image:: ../../images/hour_sin3.png

An intuitive way to show the new representation is to plot the sine vs the cosine
transformation of the hour. It will show as a 24 hour clock, and now, the distance
between two points corresponds to the difference in time as we would expect from a
24-hour cycle.

.. code:: python

    fig, ax = plt.subplots(figsize=(7, 5))
    sp = ax.scatter(df["hour_sin"], df["hour_cos"], c=df["hour"])
    ax.set(
        xlabel="sin(hour)",
        ylabel="cos(hour)",
    )
    _ = fig.colorbar(sp)

.. image:: ../../images/hour_sin4.png

We hope that cleared things up a bit.

More details
^^^^^^^^^^^^

You can find more details on to use the :class:`CyclicalFeatures()` in the
following Jupyter notebooks.

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/creation/CyclicalFeatures.ipynb>`_
