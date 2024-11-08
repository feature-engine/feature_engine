.. _information_value:

.. currentmodule:: feature_engine.selection

SelectByInformationValue
========================

:class:`SelectByInformationValue()` selects features based on whether the feature's information value score is
greater than the threshold passed by the user.

The IV is calculated as:

.. math::

   IV = ∑ (fraction of positive cases - fraction of negative cases) * WoE

where:

- the fraction of positive cases is the proportion of observations of class 1, from the total class 1 observations.
- the fraction of negative cases is the proportion of observations of class 0, from the total class 0 observations.
- WoE is the weight of the evidence.

The WoE is calculated as:

.. math::

   WoE = ln(fraction of positive cases / fraction of negative cases)

Information value (IV) is used to assess a feature's predictive power of a binary-class dependent
variable. To derive a feature's IV, the weight of evidence (WoE) must first be calculated for each
unique category or bin that comprises the feature. If a category or bin contains a large percentage
of true or positive labels compared to the percentage of false or negative labels, then that category
or bin will have a high WoE value.

Once the WoE is derived, :class:`SelectByInformationValue()` calculates the IV for each variable.
A variable's IV is essentially the weighted sum of the individual WoE values for each category or bin
within that variable where the weights incorporate the absolute difference between the
numerator and denominator. This value assesses the feature's predictive power in capturing the binary
dependent variable.

The table below presents a general framework for using IV to determine a variable's predictive power:

.. list-table::
    :widths: 30 30
    :header-rows: 1

    * - Information Value
      - Predictive Power
    * - < 0.02
      - Useless
    * - 0.02 to 0.1
      - Weak
    * - 0.1 to 0.3
      - Medium
    * - 0.3 to 0.5
      - Strong
    * - > 0.5
      - Suspicious, too good to be true

Table taken from `listendata <https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html>`_.


Example
-------
Let's see how to use this transformer to select variables from UC Irvine's credit approval data set which can
be found `here`_. This dataset concerns credit card applications. All attribute names and values have been changed
to meaningless symbols to protect confidentiality.

The data is comprised of both numerical and categorical data.

.. _here: https://archive-beta.ics.uci.edu/ml/datasets/credit+approval

Let's import the required libraries and classes:

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from feature_engine.selection import SelectByInformationValue

Let's now load and prepare the credit approval data:

.. code:: python

    # load data
    data = pd.read_csv('crx.data', header=None)

    # name variables
    var_names = ['A' + str(s) for s in range(1,17)]
    data.columns = var_names
    data.rename(columns={'A16': 'target'}, inplace=True)

    # preprocess data
    data = data.replace('?', np.nan)
    data['A2'] = data['A2'].astype('float')
    data['A14'] = data['A14'].astype('float')
    data['target'] = data['target'].map({'+':1, '-':0})

    # drop rows with missing data
    data.dropna(axis=0, inplace=True)

    data.head()

Let's now review the first 5 rows of the dataset:

.. code:: python

      A1     A2     A3 A4 A5 A6 A7    A8 A9 A10  A11 A12 A13    A14  A15  target
    0  b  30.83  0.000  u  g  w  v  1.25  t   t    1   f   g  202.0    0       1
    1  a  58.67  4.460  u  g  q  h  3.04  t   t    6   f   g   43.0  560       1
    2  a  24.50  0.500  u  g  q  h  1.50  t   f    0   f   g  280.0  824       1
    3  b  27.83  1.540  u  g  w  v  3.75  t   t    5   t   g  100.0    3       1
    4  b  20.17  5.625  u  g  w  v  1.71  t   f    0   f   s  120.0    0       1


Let's now split the data into train and test sets:

.. code:: python

    # separate train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['target'], axis=1),
        data['target'],
        test_size=0.2,
        random_state=0)

    X_train.shape, X_test.shape

We see the size of the datasets below.

.. code:: python

    ((522, 15), (131, 15))

Now, we set up :class:`SelectByInformationValue()`. We will pass six categorical
variables to the parameter :code:`variables`. We will set the parameter :code:`threshold`
to `0.2`. We see from the above mentioned table that an IV score of 0.2 signifies medium
predictive power.

.. code:: python

    sel = SelectByInformationValue(
        variables=['A1', 'A6', 'A9', 'A10', 'A12', 'A13'],
        threshold=0.2,
    )

    sel.fit(X_train, y_train)

With :code:`fit()`, the transformer:

 - calculates the WoE for each variable
 - calculates the the IV for each variable
 - identifies the variables that have an IV score below the threshold

In the attribute :code:`variables_`, we find the variables that were evaluated:

.. code:: python

    ['A1', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']

In the attribute :code:`features_to_drop_`, we find the variables that were not selected:

.. code:: python

    sel.features_to_drop_

    ['A1', 'A12', 'A13']

The attribute :code:`information_values_` shows the IV scores for each variable.

.. code:: python

   {'A1': 0.0009535686492270659,
    'A6': 0.6006252129425703,
    'A9': 2.9184484098456807,
    'A10': 0.8606638171665587,
    'A12': 0.012251943759377052,
    'A13': 0.04383964979386022}

We see that the transformer correctly selected the features that have an IV score greater
than the :code:`threshold` which was set to 0.2.

The transformer also has the method `get_support` with similar functionality to Scikit-learn's
selectors method. If you execute `sel.get_support()`, you obtain:

.. code:: python

    [False, True, True, True, True, True, True,
     True, True, True, True, False, False, True,
     True]

With :code:`transform()`, we can go ahead and drop the features that do not meet the threshold:

.. code:: python

    Xtr = sel.transform(X_test)

    Xtr.head()

.. code:: python

            A2     A3 A4 A5  A6 A7      A8 A9 A10  A11    A14  A15
    564  42.17   5.04  u  g   q  h  12.750  t   f    0   92.0    0
    519  39.17   1.71  u  g   x  v   0.125  t   t    5  480.0    0
    14   45.83  10.50  u  g   q  v   5.000  t   t    7    0.0    0
    257  20.00   0.00  u  g   d  v   0.500  f   f    0  144.0    0
    88   34.00   4.50  u  g  aa  v   1.000  t   f    0  240.0    0


Note that :code:`Xtr` includes all the numerical features - i.e., A2, A3, A8, A11, and A14 - because
we only evaluated a few of the categorical features.

And, finally, we can also obtain the names of the features in the final transformed dataset:

.. code:: python

    sel.get_feature_names_out()

    ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A14', 'A15']


If we want to select from categorical and numerical variables, we can do so as well by
sorting the numerical variables into bins first. Let's sort them into 5 bins of equal-frequency:

.. code:: python

    sel = SelectByInformationValue(
        bins=5,
        strategy="equal_frequency",
        threshold=0.2,
    )

    sel.fit(X_train.drop(["A4", "A5", "A7"], axis=1), y_train)

If we now inspect the information values:

.. code:: python

   sel.information_values_

We see the following:

.. code:: python

    {'A1': 0.0009535686492270659,
     'A2': 0.10319123021570434,
     'A3': 0.2596258749173557,
     'A6': 0.6006252129425703,
     'A8': 0.7291628533346297,
     'A9': 2.9184484098456807,
     'A10': 0.8606638171665587,
     'A11': 1.0634602064399297,
     'A12': 0.012251943759377052,
     'A13': 0.04383964979386022,
     'A14': 0.3316668794040285,
     'A15': 0.6228678069374612}

And if we inspect the features to drop:

.. code:: python

   sel.features_to_drop_

We see the following:

.. code:: python

    ['A1', 'A2', 'A12', 'A13']


Note
----

The WoE is given by a logarithm of a fraction. Thus, if for any category or bin, the fraction of
observations of class 0 is 0, the WoE is not defined, and the transformer will raise an error.

If you encounter this problem try grouping variables into fewer bins if they are numerical,
or grouping rare categories with the RareLabelEncoder if they are categorical.

Additional resources
--------------------

For more details about this and other feature selection methods check out these resources:


.. figure::  ../../images/fsml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-selection-for-machine-learning

   Feature Selection for Machine Learning

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

.. figure::  ../../images/fsmlbook.png
   :width: 200
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-selection-in-machine-learning-book

   Feature Selection in Machine Learning

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
|

Both our book and course are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.