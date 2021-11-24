.. _psi_selection:

.. currentmodule:: feature_engine.selection

DropHighPSIFeatures
===================

The :class:`DropHighPSIFeatures()` finds and removes features with unstable distribution
from a pandas dataframe.
The stability of the distribution is computed using the `Population Stability
Index (PSI)` and
all features having a PSI value above a given threshold are removed.

Unstable features may introduce an additional bias in a model if the training population
significantly differs from the population in production. Removing these features leads to
more robust models and therefore to better performances over time. In the field of Credit Risk
modelling, eliminating features with high PSI is common practice and usually required by the
Regulator.

##### Splitting the input dataframe

Computing the PSI involves comparing two distributions. In
:class: `DropHighPSIFeatures()` the input dataframe is split in
two parts (called base and test); these two parts are used to calculate the
PSI value for each feature.


The class offers three ways to split the input dataframe:

- Using proportions: For example 75% and 25%.
- Using a cut-off value: Up-to and above the cut-off value.
- Observations belonging to a group (defined by a list of values) and
  observations not belonging to that group.

The split can be done based on two dimensions:

- The index (i.e. its value).
- A specified column.


Procedure
---------

:class:`DropHighPSIFeatures()` works according to the following procedure:

The features in scope for the selector are identified. It is either the columns
of the dataframe whose labels are passed in the *variables* argument or all
numerical columns from the dataframe if the *variables* argument is not explicitly defined.


The input dataframe is split in two subparts according to the parameters passed
to the `init` method. The method used to split the dataframe performs the following
steps:
- Identify the reference as the column to use for splitting the dataframe. It is either
a column provided by the user or the index of the dataframe.
- Determine the cut-off value. The cut-off is either directly provided by the user as
argument or it is computed based on the proportion (i.e. what is the cut-off value
if we split the dataframe on a X, 1-X basis).
- Define the base as the dataframe containing all observations associated with values
of the reference up to the cut-off value. The test dataframe contains all other
observations.
- If the cut-off value is defined as a list, the split is done in a slightly different
way. The base contains all observations for which the reference is part of the list.
The test dataframe contains all other observations.

Then for each feature in scope:
- Discretize the values using the binning strategy defined by the user.
Two binning strategies can be used: equal frequency (i.e. each bin contains
the same number of observations) or equal width (i.e. the difference between
the upper and the lower limits of the bin is the same for all bins).
The bins are defined on the values of the base dataframe.
- Determine percentage of observations per bin for the base and the test
dataframes.
- Compute the PSI value based on the percentage of observations per bin.
- If the PSI is above the defined threshold, add the feature to the list of features to drop.


##### Remarks on the use of the PSI.

When working with PSI, it is worth highlighting the following:

The PSI is not symmetric; switching the order of the basis and test dataframes
in the PSI calculation will lead to different values.

The number of bins used to define the distributions has an impact on the PSI values.

The PSI is a suitable metric for numerical features (i.e., either continuous
or with high cardinality).

For categorical or discrete features, the change in distributions is better
assessed with Chi-squared.

Different thresholds can be used to assess the magnitude of the distribution
shift according to the PSI value. The most commonly used thresholds are:
below 10% (the variable has not experienced a significant shift) and above 25%
(the variable has experienced a major shift). 
Between those two values, the shift is intermediate.


Examples
--------

The complexity of the class lies in the different options to split
the input dataframe in order to compute the PSI. Therefore several examples,
illustrating the different approaches, are provided below.

**Case 1: split data based on proportions (split_frac parameter)**

Let's first define a toy dataframe containing random variables as basis
for the PSI calculations.

.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from feature_engine.selection import DropHighPSIFeatures

    # Define a dataframe with 200 observations from 6 random variables
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        random_state=0
    )

    colnames = ["var_" + str(i) for i in range(n_feat)]
    X = pd.DataFrame(X, columns=colnames)

The default approach in :class:`DropHighPSIFeatures()` is to split the
input dataframe (X) in two equally sized parts based on the value of the
index. By passing a value to the `split_frac` argument, the ratio between
the sizes of the two parts can be adjusted.

.. code:: python

    # Remove the features with high PSI values using a 60-40 split.

    transformer = DropHighPSIFeatures(split_frac=0.6)
    X_transformed = transformer.fit_transform(X)

- The value of the split_frac argument (0.6) means that the two dataframes used
  to compute the PSI values (base and test) will be split according to a 60% - 40% basis.
- The fit method performs the split of the dataframe and the calculations of 
  procedure described above. The PSI values are accessible through the `.psi_values_` attribute.

.. code:: python

    transformer.psi_values_

The analysis of the PSI values (see below) shows that only feature 3 (called `var_3`)
has a PSI above the 0.25 threshold (default value) and will be removed
by the transform method.

.. code:: python

    {'var_0': 0.10200882787259648,
    'var_1': 0.06247480220678372,
    'var_2': 0.231106813775744,
    'var_3': 0.2662638025200693,
    'var_4': 0.19861346887805775,
    'var_5': 0.1411194164512627}

The cut-off value used to split the dataframe is stored in the
`DropHighPSIFeatures` object. It can be
accessed via the `.cut_off_` attribute:

.. code:: python

    transformer.cut_off_

The value of 119.4 means that observations with index from 0 to 119 are used
to define the
base dataframe. This corresponds to 60% (120 / 200) of the original dataframe
(X).

The base and the test dataframes are not directly accessible. However they can
be (re-)computed
using the `.split_dataframe()` method.

.. code:: python

    base, test = transformer._split_dataframe(X)


**Case 2: split data based on variable (cut_off is numerical value)**

:class:`DropHighPSIFeatures()` allows to define the column used to
split the dataframe. Two options are then available to the user:
- Split by proportion. This is an approach similar to the one described in the
first use case.
- Split by threshold. Using the `cut_off` argument, the user can define the
specific threshold for the split.

A real life example for this case is the use of the customer ID or contract ID
to split the dataframe. These ID's are often increasing over time which justify
their use to assess population shift in the features.

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

:class:`DropHighPSIFeatures()` is called in such a way that the base dataframe
used in the calculation of the PSI values contains all observations with `var_1`
lower or equal to 0.5. The test dataframe contains all other values.

This is shown by inspecting the dataframe using the `split_dataframe` method.

.. code:: python

    base, test = transformer._split_dataframe(X)
    base.describe()

The maximum value for `var_1` column of the base dataframe is just below the
cut-off
value of 0.5.

.. code:: python

    test.describe()

The minimum value for the `var_1` column of the test dataframe is above the 0.5
cut-off.

When looking at the PSI values:

.. code:: python

    transformer.psi_values_

We see that :class:`DropHighPSIFeatures()` does not provide any PSI value for
the `var_1` feature.

.. code:: python

    {'var_0': 0.19387083701883132,
    'var_2': 0.12789758898627593,
    'var_3': 0.20928408122831613,
    'var_4': 0.3614010092217242,
    'var_5': 0.17200356108416925}

This is the expected behaviour as the column used to split the dataframe is
further excluded from the calculations. Based on the
definition of the PSI, it
does not make sense to compute the PSI value for the column defined in
`split_col`


**Case 3: split data based on variable (cut_off is date)**


:class:`DropHighPSIFeatures()` can handle different types of `split_col`
variables. In the following example
it is shown how it works with a date.

This case is representative when investigating population shift after an
event like the start of the Covid-19 pandemic.

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

    # Add a date to the dataframe

    X['time'] = [date(year, 1, 1) for year in range(1000, 2000)]

Performing the PSI elimination by considering two periods of time is done simply
by providing a date as cut-off value. In the example below the PSI calculations
will be done comparing the period up to the French revolution and after.


.. code:: python

    transformer = DropHighPSIFeatures(split_col='time', cut_off=date(1789, 7, 14))

To check if the split is performed as expected, we look at the date
for the base and test dataframes coming from the `_split_dataframe()` method.

.. code:: python

    base, test = transformer._split_dataframe(X)
    print(base.time.max(), test.time.min())

This yields the following result.
.. code:: python

    1789-01-01 1790-01-01


**Case 4: split data based on variable (cut_off is list)**

:class:`DropHighPSIFeatures()` can also split the original dataframe based on
a string variable. The cut-off can then be defined in two ways:
- Using a single string.
- Using a list of values.

In the first case, the column with the categorical variable is
sorted alphabetically and the split is determined by the cut-off. We advise
the user to be very cautious when working in such a setting as alphabetical
sorting in combination with a cut-off does not always provide obvious results.

A real life example for this case is the computation of the PSI between
different customer segments like 'Retail', 'SME' and 'Wholesale'.

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

We can define a simple cut-off value (for example the letter C).

.. code:: python

    transformer = DropHighPSIFeatures(split_col='group', cut_off='C')
    X_no_drift = transformer.fit_transform(X)

In order to understand how the dataframe is split to compute the PSI,
we look at the output of the `_split_dataframe` method that is called
during the PSI calculations.

.. code:: python

    base, test = transformer._split_dataframe(X)
    print(base.group.unique())
    print(test.group.unique())

This yields the following results:

.. code:: python

    ['A' 'B' 'C']
    ['D' 'E']

The other option considered in this case is when `cut_off` is defined as
a list.

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

Running `DropHighPSIFeatures` in done is a similar way as in the previous cases.

.. code:: python

    transformer = DropHighPSIFeatures(split_col='group', cut_off=['A', 'C', 'E'])
    base, test = transformer._split_dataframe(X)

According to the parameters passed when initializing the `DropHighPSIFeatures`
object,
we expect the base dataframe to contain all observations associated with the groups
A, C and E and the test dataframe to contain all observations associated with the groups
B and D. This is exactly what happens.

.. code:: python

    base, test = transformer._split_dataframe(X)
    print(base.group.unique())
    print(test.group.unique())

This yields the following results:

.. code:: python

    ['A' 'B' 'C']
    ['D' 'E']

Note the defining a list works with all type of data (string, date, integer and float)
that :class:`DropHighPSIFeatures()` can handle.:class:`DropHighPSIFeatures()`


**Case 5: split data based on unique values (split_distinct)**

A variant to the previous example is the use of the split_distinct functionality.
In that case, the split is not done based on the number observations from
`split_col` but from the number of distinct values in `split_col`.

A real life example for this case is when dealing with groups of different sizes
like customer's incomes classes ('1000', '2000', '3000', '4000', ...).
split_distinct allows to control the numbers of classes in the base and test
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

Now the `group` column contains 500 observations in the (A, B, C, D, E)
group and 500 in the (F) group. This is reflected in the output of
`DropHighPSIFeatures` when used with the default parameter values.

.. code:: python

    transformer = DropHighPSIFeatures(split_col='group')
    base, test = transformer._split_dataframe(X)
    print(base.group.unique(), base.shape)
    print(test.group.unique(), test.shape)

That yields the following output:

.. code:: python

    ['A' 'B' 'C' 'D' 'E'] (500, 8)
    ['F'] (500, 8)

If we pass the `split_distinct=True` argument when initialize
the `DropHighPSIFeatures` object, the split will

.. code:: python

    transformer = DropHighPSIFeatures(split_col='group', split_distinct=True)
    base, test = transformer._split_dataframe(X)
    print(base.group.unique(), base.shape)
    print(test.group.unique(), test.shape)

That yields the following output:

.. code:: python

    ['A' 'B' 'C'] (300, 8)
    ['D' 'E' 'F'] (700, 8)

More details
^^^^^^^^^^^^

In this notebook, we show how to use :class:`DropHighPSIFeatures()`.

If we detail very well how to use all parameters here, we may not need a notebook.
Notebooks are located here:

https://github/feature-engine/feature-engine-examples/blob/main/selection/
