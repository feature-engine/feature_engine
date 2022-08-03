.. _mean_discretiser:

.. currentmodule:: feature_engine.discretisation

MeanDiscretiser
===============

The :class:`MeanDiscretiser()` sorts numerical variables into bins of equal-width or
equal-frequency and then returns the mean value of the target per interval.

Under the hood, :class:`MeanDiscretiser()` uses the :class:`EqualFrequencyDiscretiser()`
or :class:`EqualWidthDiscretiser()` for the discretization into intervals. Once the numerical
variables are separated into bins, the :class:`MeanEncoder()` replaces the interval limits
with the mean of the target per bin interval. The number of bins is determined by the user.

**Example**

Let's look at an example using the California Housing Dataset.

First, let's load the data and separate it into train and test:

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    from feature_engine.discretisation import MeanDiscretiser

    # Load dataset
    california_dataset = fetch_california_housing()
    data = pd.DataFrame(california_dataset.data, columns=california_dataset.feature_names)

    # Seperate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, california_dataset["target"], test_size=0.3,
        random_state=0,
    )

Now, we set up the :class:`MeanDiscretiser()` to sort the 3 indicated variables into 5
intervals, and then return the mean target value per interval:

.. code:: python

    # set up the discretisation transformer
    disc = MeanDiscretiser(variables=["HouseAge", "AveRooms", "Population"],
                           strategy="equal_frequency",
                           bins=5,
                           )

    # fit the transformer
    disc.fit(X_train, y_train)

With `fit()` the transformer learns the boundaries of each interval, and the target mean
value for each interval, which are stored in `binner_dict_` parameter:

.. code:: python

    disc.binner_dict_

The `binner_dict_` contains the mean value of the target per interval, per variable.
So we can easily use this dictionary to map the numbers to the discretised bins.

# TODO: after adding the binner_dict_ parameter in the main class, we need to replace this output:

.. code:: python


    {'HouseAge': {Interval(-inf, 17.0, closed='right'): 2.0806529160739684,
        Interval(17.0, 25.0, closed='right'): 2.097539197771588,
        Interval(25.0, 33.0, closed='right'): 2.0686614742967993,
        Interval(33.0, 40.0, closed='right'): 2.1031412685185185,
        Interval(40.0, inf, closed='right'): 2.0266248845381525},
    'AveRooms': {Interval(-inf, 4.281, closed='right'): 2.0751556984478934,
        Interval(4.281, 4.94, closed='right'): 2.0353196247563354,
        Interval(4.94, 5.524, closed='right'): 2.122038111675127,
        Interval(5.524, 6.258, closed='right'): 2.0422810965372507,
        Interval(6.258, inf, closed='right'): 2.103166361757106},
    'Population': {Interval(-inf, 709.0, closed='right'): 2.0853869883779685,
        Interval(709.0, 1004.0, closed='right'): 2.0658340239808153,
        Interval(1004.0, 1346.0, closed='right'): 2.0712619255907487,
        Interval(1346.0, 1905.0, closed='right'): 2.0454417591204397,
        Interval(1905.0, inf, closed='right'): 2.108366283914729}}

We can now go ahead and replace the numerical variables with the target mean values:

.. code:: python

    # transform the data
    train_t = disc.transform(X_train)
    test_t = disc.transform(X_test)

