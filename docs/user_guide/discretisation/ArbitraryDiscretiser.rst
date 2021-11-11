.. _arbitrary_discretiser:

.. currentmodule:: feature_engine.discretisation

ArbitraryDiscretiser
====================

The :class:`ArbitraryDiscretiser()` sorts the variable values into contiguous intervals
which limits are arbitrarily defined by the user. Thus, you must provide a dictionary
with the variable names as keys and the limits of the intervals in a list as values,
when setting up the discretiser.

The :class:`ArbitraryDiscretiser()` works only with numerical variables. The discretiser
will check that the variables entered by the user are present in the train set and cast
as numerical.

**Example**

Let's take a look at how this transformer works. First, let's load a dataset and plot a
histogram of a continuous variable. We use the boston house prices dataset that comes
with Scikit-learn.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_boston
    from feature_engine.discretisation import ArbitraryDiscretiser

    boston_dataset = load_boston()
    data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

    data['LSTAT'].hist(bins=20)
    plt.xlabel('LSTAT')
    plt.ylabel('Number of observations')
    plt.title('Histogram of LSTAT')
    plt.show()

.. image:: ../../images/lstat_hist.png

Now, let's discretise the variable into arbitrarily determined intervals. We want the
interval names as integers, so we set `return_boundaries` to False.

.. code:: python

    user_dict = {'LSTAT': [0, 10, 20, 30, np.Inf]}

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_object=False, return_boundaries=False)

    X = transformer.fit_transform(data)

Now, we can go ahead and plot the variable after the transformation:

.. code:: python

    X['LSTAT'].value_counts().plot.bar()
    plt.xlabel('LSTAT - bins')
    plt.ylabel('Number of observations')
    plt.title('Discretised LSTAT')
    plt.show()

.. image:: ../../images/lstat_disc_arbitrarily.png

Note that in the above figure the intervals are represented by digits.

Alternatively, we can return the interval limits in the discretised variable by
setting `return_boundaries` to True.

.. code:: python

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_object=False, return_boundaries=True)
    X = transformer.fit_transform(data)

    X['LSTAT'].value_counts().plot.bar(rot=0)
    plt.xlabel('LSTAT - bins')
    plt.ylabel('Number of observations')
    plt.title('Discretised LSTAT')
    plt.show()

.. image:: ../../images/lstat_disc_arbitrarily2.png

**Discretisation plus encoding**

If we return the interval values as integers, the discretiser has the option to return
the transformed variable as integer or as object. Why would we want the transformed
variables as object?

Categorical encoders in Feature-engine are designed to work with variables of type
object by default. Thus, if you wish to encode the returned bins further, say to try and
obtain monotonic relationships between the variable and the target, you can do so
seamlessly by setting `return_object` to True. You can find an example of how to use
this functionality `here <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/discretisation/ArbitraryDiscretiser_plus_MeanEncoder.ipynb>`_.

More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/discretisation/ArbitraryDiscretiser.ipynb>`_
- `Jupyter notebook - Discretiser plus Mean Encoding <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/discretisation/ArbitraryDiscretiser_plus_MeanEncoder.ipynb>`_
