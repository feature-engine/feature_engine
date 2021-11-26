.. _psi_selection:

.. currentmodule:: feature_engine.selection

DropHighPSIFeatures
===================

The :class:`DropHighPSIFeatures()` finds and removes features with changes in their
distribution, i.e. "unstable values", from a pandas dataframe.
The stability of the distribution is computed using the **Population Stability
Index (PSI)** and all features having a PSI value above a given threshold are removed.

Unstable features may introduce an additional bias in a model if the training population
significantly differs from the population in production. Removing features for
which a shift in the distribution is suspected leads to
more robust models and therefore to better performance. In the field of Credit Risk
modelling, eliminating features with high PSI is common practice and usually required by the
Regulator.

Population Stability Index - PSI
--------------------------------

The PSI is a measure of how much a population has changed in time or how different the distributions
are between two different population samples.

To determine the PSI, continuous features are sorted into discrete intervals, the
fraction of observations per interval is then determined, and finally those values
are compared between the 2 groups, or as we call them in Feature-engine, between the
basis and test sets, to obtain the PSI.

In other words, the PSI is computed as follows:

- Define the intervals into which the observations will be sorted.
- Sort the feature values into those intervals.
- Determine the fraction of observations within each interval.
- Compute the PSI.

The PSI is determined as:

.. math::

    PSI = \sum_{i=1}^n (test_i - basis_i) . ln(\frac{test_i}{basis_i})

where `basis` and `test` are the reference and comparison datasets, respectively, and `i`
refers to the interval.

In other words, the PSI determines the difference in the proportion of observations in each
interval, between the reference (aka, original) and comparison datasets.

In the PSI equation, `n` is the total number of intervals.

Important
~~~~~~~~~

When working with the PSI it is worth highlighting the following:

- The PSI is not symmetric; switching the order of the basis and test dataframes in the PSI calculation will lead to different values.
- The number of bins used to define the distributions has an impact on the PSI values.
- The PSI is a suitable metric for numerical features (i.e., either continuous or with high cardinality).
- For categorical or discrete features, the change in distributions is better assessed with Chi-squared.

Threshold
~~~~~~~~~

Different thresholds can be used to assess the magnitude of the distribution shift according
to the PSI value. The most commonly used thresholds are:

- Below 10%, the variable has not experienced a significant shift.
- Above 25%, the variable has experienced a major shift.
- Between those two values, the shift is intermediate.


Procedure
---------

To compute the PSI, the :class:`DropHighPSIFeatures()` splits the input dataset in
two: a basis data set (aka the reference data) and a test set. The basis data set is assumed to contain
the expected or original feature distributions. The test set will be assessed
against the basis data set.

In the next step, the interval boundaries are determined based on the features in the basis
or reference data. These intervals can be determined to be of equal with, or equal number
of observations.

Next, :class:`DropHighPSIFeatures()` sorts each of the variable values into those intervals, both in the
basis and test datasets, and then determines the proportion (percentage) of observations
within each interval.

Finally, the PSI is determined as indicated in the previous paragraph for each indicated feature.
With the PSI value per feature, :class:`DropHighPSIFeatures()` can now select the features that are unstable and
drop them, based on a threshold.


Splitting the data
------------------

:class:`DropHighPSIFeatures()` allows us to determine how much a feature distribution has
changed in time, or how much it differs between 2 groups.

If we want to evaluate the distribution change in time, we can use a datetime variable as splitting
reference and provide a datetime cut-off as split point.

If we want to compare the distribution change between 2 groups, :class: `DropHighPSIFeatures()`
offers 3 different approaches to split the input dataframe:

- Proportion of observations.
- Proportions of unique observations.
- Using a cut-off value.


Proportion of observations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Splitting by proportion of observations will result in a certain proportion of observations
allocated to either the reference and test datasets. For example, if we set `split_frac=0.75`,
then 75% and 25% of the observations will be put into the reference and test data, respectively.

If we select this method, we can pass a variable in the parameter `split_col` or leave it to None.

Note that this method **does not shuffle** the dataset. This means that if `split_frac=0.75`, the
first 75% of the rows will be allocated to the reference set, and the bottom 25% to the test set.

If the rows in your dataset are sorted in time, this could be a good option to split the
dataframe.

Proportions of unique observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we split based on proportion of unique observations, it is important that we indicate which
column we want to use as reference in the `split_col` parameter, to make a meaningful split.

:class:`DropHighPSIFeatures()` will first identify the unique values of the variable in
`split_col`. Then it will put a certain proportion of those values into the reference
dataset and the remaining to the test dataset. The proportion is indicated in the parameter
`split_frac`.

This split makes sence when we have for example unique customer identifiers, and multiple rows
per customer in the dataset. We want to make sure that all rows belonging to the same customer
are allocated either in the reference or test data, but not both. And we want to make sure that
we have a certain percentage of customers in either dataframe.

Thus, if `split_frac=0.6` and `split_distinct=True`, :class:`DropHighPSIFeatures()` will send
the first 60% of customers to the reference dataset, and the bottom 40% to the test set. And it will
ensure that rows beloging to the same customer are just in one of the 2 dataframes.

Using a cut-off value
~~~~~~~~~~~~~~~~~~~~~

We have the option to pass a reference variable to use to split the dataframe using `split_col` and
also a cut-off value in the `cut_off` parameter. The cut-off value can be a number, integer or float,
a date or a list of values.

If we pass a datetime column in `split_col` and a datetime value in the `cut_off`, we can split the
data in a temporal manner. Observations collected before the time indicated will be sent to the reference
dataframe, and the remaining to the test set.

If we pass a list of values in the `cut_off` all observations whcih values are included in the
list go into the reference dataframe, and the remaining to the test dataframe. This split is useful
if we have a categorical variable indicating a portfolio from which the observations have been collected.
For example, if we set `split_col=portfolio` and `cut_off=['port_1`, 'port_2']`, all observations
that belong to the first and second portolio will be sent to the reference dataset, and the observations
from other portfolios to the test set.

Finally, if we pass a number to `cut_off`, all observations which value in the variable indicated
in `split_col` is below the cut-off, will be sent to the reference data, alternatively to the test data.
#TODO: can we think of an example??

split_col
~~~~~~~~~

To split the dataset, we recommend that you indicate which column you want to use as
reference in the `split_col` parameter. If you don't, the split will be done based on the
values of the dataframe index. This might be a good option if the index contains meaningful
values or if splitting just based on `split_frac`.


Examples
--------

The versatility of the class lies in the different options to split the input dataframe
in a reference or basis data set with the "expected" distributions and a test set, which
will be evaluated against the reference.

After splitting the data, :class:`DropHighPSIFeatures()` goes ahead and compares the
feature distributions in both datasets by computing the PSI.

To illustrate how to best use this class depending on your scenario or data, we provide
various examples illustrating the different approaches or scenarios.

Case 1: split data based on proportions (split_frac)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, :class:`DropHighPSIFeatures()` will split the dataset in 2, based on the
indicated proportion. The proportion is indicated in the `split_frac` parameter. You have
the option to select a variable in `split_col` or leave it to None.

Let's first create a toy dataframe containing random variables.

.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from feature_engine.selection import DropHighPSIFeatures

    # Create a dataframe with 200 observations and 6 random variables
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        random_state=0
    )

    colnames = ["var_" + str(i) for i in range(n_feat)]
    X = pd.DataFrame(X, columns=colnames)

The default approach in :class:`DropHighPSIFeatures()` is to split the
input dataframe `X` in two equally sized data sets. You can adjust the proportions by changing
the value in the `split_frac` parameter.

For example, let's split the input dataframe into a reference data set containing 60% of
the observations and a test set containing 40% of the observations.

.. code:: python

    # Remove the features with high PSI values using a 60-40 split.

    transformer = DropHighPSIFeatures(split_frac=0.6)
    X_transformed = transformer.fit_transform(X)

The value of `split_frac` tells :class:`DropHighPSIFeatures()` to split X according to a
60% - 40% ratio. The `fit()` method performs the split of the dataframe and the calculation
of the PSI.

The PSI values are accessible through the `psi_values_` attribute:

.. code:: python

    transformer.psi_values_

The analysis of the PSI values (see below) shows that only feature 3 (called `var_3`)
has a PSI above the 0.25 threshold (default value) and will be removed
by the `transform` method.

.. code:: python

    {'var_0': 0.10200882787259648,
    'var_1': 0.06247480220678372,
    'var_2': 0.231106813775744,
    'var_3': 0.2662638025200693,
    'var_4': 0.19861346887805775,
    'var_5': 0.1411194164512627}

The cut-off value used to split the dataframe is stored in the `cut_off_` attribute:

.. code:: python

    transformer.cut_off_

#TODO: can you add the output of the previous command please?

The value of 119.4 means that observations with index from 0 to 119 are used
to define the basis dataframe. This corresponds to 60% (120 / 200) of the original dataframe
(X).

Splitting with proportions will order the index or the reference column first, and then
determine the data that will go into each dataframe. In other words, the order of the index
or the variable indicated in `split_col` matters. Observations with the lowest values will
be send to the basis dataframe and the ones with the highest values to the test set.


Case 2: split data based on variable (numerical cut_off)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`DropHighPSIFeatures()` allows your to define the column used to
split the dataframe. There are two options available:

- Split by proportion: This approach is similar to the one described in the previous example.
- Split by threshold: Using the `cut_off` argument, the user can define the specific threshold for the split.

A real life example for this case is the use of the customer ID or contract ID
to split the dataframe. These ID's are often increasing in value over time which justify
their use to assess distribution shift in the features.

Let's create a toy dataframe with random variables.

.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from feature_engine.selection import DropHighPSIFeatures

    X, y = make_classification(
            n_samples=200,
            n_features=6,
            random_state=0
        )

    colnames = ["var_" + str(i) for i in range(n_feat)]
    X = pd.DataFrame(X, columns=colnames)

    # Call the PSI selector
    transformer = DropHighPSIFeatures(split_col='var_1', cut_off=0.5)
    X_transformed = transformer.fit_transform(X)

In this case, :class:`DropHighPSIFeatures()` will allocate in the basis or reference data
set, all observations which values in `var_1` are <= 0.5. The test dataframe contains the
remaining observations.

The method `fit()` will determine the PSI values, which are stored in the class:

.. code:: python

    transformer.psi_values_

We see that :class:`DropHighPSIFeatures()` does not provide any PSI value for
the `var_1` feature, because this variable was used as a reference to split the data.

.. code:: python

    {'var_0': 0.19387083701883132,
    'var_2': 0.12789758898627593,
    'var_3': 0.20928408122831613,
    'var_4': 0.3614010092217242,
    'var_5': 0.17200356108416925}


Case 3: split data based on time (date as cut_off)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


:class:`DropHighPSIFeatures()` can handle different types of `split_col`
variables. The following case illustrates how it works with a date variable. In fact,
we often want to determine if the distribution of a feature changes in time, or after a
certain event like the start of the Covid-19 pandemic.

This is how to do it. Let's create a toy dataframe with random numerical variables and a
date variable.

.. code:: python

    import pandas as pd
    from datetime import date
    from sklearn.datasets import make_classification
    from feature_engine.selection import DropHighPSIFeatures

    X, y = make_classification(
            n_samples=1000,
            n_features=6,
            random_state=0
        )

    colnames = ["var_" + str(i) for i in range(n_feat)]
    X = pd.DataFrame(X, columns=colnames)

    # Add a date variable to the dataframe
    X['time'] = [date(year, 1, 1) for year in range(1000, 2000)]

Dropping features with high PSI values comparing two periods of time is done simply
by providing the name of the column with the date and a cut-off date.

In the example below the PSI calculations
will be done comparing the periods up to the French revolution and after.



Case 4: split data based on a categorical variable (list as cut_off)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`DropHighPSIFeatures()` can also split the original dataframe based on
a categorical variable. The cut-off can then be defined in two ways:

- Using a single string.
- Using a list of values.

In the first case, the column with the categorical variable is
sorted alphabetically and the split is determined by the cut-off. We advise
the user to be very cautious when working in such a setting as alphabetical
sorting in combination with a cut-off does not always provide obvious results.

A better way of using this class would be to pass a list with the values of the variable
that want to be sent to the reference dataframe.

A real life example for this case is the computation of the PSI between
different customer segments like 'Retail', 'SME' or 'Wholesale'.

Let's show how to set up the transformer in this case.

.. code:: python

    import pandas as pd

    from sklearn.datasets import make_classification
    from feature_engine.selection import DropHighPSIFeatures

    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        random_state=0
    )

    colnames = ["var_" + str(i) for i in range(n_feat)]
    X = pd.DataFrame(X, columns=colnames)

    # Add a categorical column
    X['group'] = ["A", "B", "C", "D", "E"] * 200

We can define a simple cut-off value (for example the letter C). In this case, observations
with values that come before C, alphabetically, will be allocated to the reference data set.

.. code:: python

    transformer = DropHighPSIFeatures(split_col='group', cut_off='C')
    X_no_drift = transformer.fit_transform(X)

The second option consists in passing a list of values to `cut_off`.

.. code:: python

    import pandas as pd

    from sklearn.datasets import make_classification
    from feature_engine.selection import DropHighPSIFeatures

    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        random_state=0
    )

    colnames = ["var_" + str(i) for i in range(n_feat)]
    X = pd.DataFrame(X, columns=colnames)

    # Add a categorical column
    X['group'] = ["A", "B", "C", "D", "E"] * 200

Now we set up :class:`DropHighPSIFeatures()` so that it allocates observations whihc values
in the variable 'group' are one of the list ['A', 'C', 'E']).

.. code:: python

    transformer = DropHighPSIFeatures(split_col='group', cut_off=['A', 'C', 'E'])
    basis, test = transformer._split_dataframe(X)


Case 5: split data based on unique values (split_distinct)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A variant to the previous example is the use of the `split_distinct` functionality.
In that case, the split is not done based on the number observations from
`split_col` but from the number of distinct values in split_col.

A real life example for this case is when dealing with groups of different sizes
like customers income classes ('1000', '2000', '3000', '4000', ...).
Split_distinct allows to control the numbers of classes in the basis and test
dataframes regardless of the number of observations in each class.


.. code:: python

    import pandas as pd

    from sklearn.datasets import make_classification
    from feature_engine.selection import DropHighPSIFeatures

    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        random_state=0
    )

    colnames = ["var_" + str(i) for i in range(n_feat)]
    X = pd.DataFrame(X, columns=colnames)

    # Add a categorical column
    X['group'] = ["A", "B", "C", "D", "E"] * 100 + ["F"] * 500

The `group` column contains 500 observations in the (A, B, C, D, E)
group and 500 in the (F) group.

If we pass the `split_distinct=True` argument when initializing
the `DropHighPSIFeatures` object, the split will ensures the basis and
the test dataframes contain the same number of unique values in the `group`
column


More details
~~~~~~~~~~~~

In this notebook, we show how to use :class:`DropHighPSIFeatures()`.

If we detail very well how to use all parameters here, we may not need a notebook.
Notebooks are located here:

https://github/feature-engine/feature-engine-examples/blob/main/selection/
