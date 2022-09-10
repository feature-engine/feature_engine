.. _information_value:

.. currentmodule:: feature_engine.selection

SelectByInformationValue
========================

:class:`SelectByInformationValue()` selects features based on whether the feature's information value is
greater than the threshold passed by the user. The transformer is only compatible with categorical
features.

Information value (IV) is used to assess a categorical feature's predictive power of a binary-class dependent
variable. To derive a feature's IV, the weight of evidence (WoE) must first be calculated for each
unique category or bin that comprises the feature. If a category or bin contains a large percentage
of true or positive labels compared to the percentage of false or negative labels, then that category
or bin will have a high WoE value.

:class:`WoE()` is used to calcaluate the WoE values for each category.

Once the WoE is derived, :class:`SelectByInformationValue()` calculates the information value (IV)
for each unique category or bin. The transformer than sums the individual IVs for each category providing
the IV score for the feature. This value assesses the feature's predictive power in capturing the binary
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


+----------------------+---------------------------------+
| Information Value    | Predictive Power                |
+======================+=================================+
| < 0.02               | Useless                         |
+----------------------+---------------------------------+
| 0.02 to 0.1          | Weak                            |
+----------------------+---------------------------------+
| 0.1 to 0.3           | Medium                          |
+----------------------+---------------------------------+
| 0.3 to 0.5           | Strong                          |
+----------------------+---------------------------------+
| > 0.5                | Suspicious, too good to be true |
+----------------------+---------------------------------+


Important
---------
:class:`SelectByInformationValue()` automatically identifies categorical variables, i.e., variable types that
are object or categorical. If any dataset's categorical variables uses numeric values for its categories or bins
the init parameter :code:`ignore_format` should be set to :code:`False`.


Example
-------
Let's see how to use this transformer to select variables from UC Irvine's credit approval data set which can
be found `here`_. This dataset concerns credit card applications. All attributes names and values have been changed
to meaningless symbols to protect confidentiality.

The data is comprised of both numerical and categorical data.

.. _here: https://archive-beta.ics.uci.edu/ml/datasets/credit+approval

Let's import the required libraries and classes:

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from feature_engine import SelectByInformationValue

Let's now load and prepare the credit approval data:

.. code:: python

    # load data
    data = pd.read_csv('data/crx.data', header=None)

    # name variables
    var_names = ['A' + str(s) for s in range(1,17)]
    data.columns = var_names
    data.rename(columns={'A16': 'target'}, inplace=True)

    # preprocess data
    data = data.replace('?', np.nan)
    data['A2'] = data['A2'].astype('float')
    data['A14'] = data['A14'].astype('float')
    data['target'] = data['target'].map({'+':1, '-':0})

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

    ((552, 15), (138, 15))

Now, we set up :class:`SelectByInformationValue()`. We will pass seven categorical variables to the init parameter
:code:`variables`. The transformer confirms that these variables exist within the dataset and that the variables are
categorical. We will set parameter code:`threshold` to `0.2`. We see from the abovementioned tables that an IV score
of 0.2 signifies medium predictive power.

.. code:: python

    sel = SelectByInformationValue(
        variables=['A1', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'],
        threshold=0.2,
        ignore_format=False,
        confirm_variables=False,
    )

    sel.fit(X_train, X_test)

With :code:`fit()` the transformer:

 - calculate the WoE for each variable
 - calculate the the IV for each variable
 - identify the variables that have an IV score below the passed threshold

In the attribute :code:`variables_` we find the variables that were evaluated:

.. code:: python

    ['A1', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']

In the attribute :code:`features_to_drop_` we find the variables that were not selected:

.. code:: python

    sel.features_to_drop_

    ['A1', 'A6', 'A10', 'A12']

The attribute :code:`information_values_` shows the IV scores for each variable. Let's look at
:code:`information_values_` with the values rounded to three decimal places:

.. code:: python

    # round IV scores to 3 decimal places
    info_vals_rnd = {k: v.round(3) for k, v in sel.information_values_.items()}
    info_val_rnd

    {'A1': -2.107,
     'A6': -3.842,
     'A7': 10.837,
     'A9': 51.237,
     'A10': -7.108,
     'A12': -0.503,
     'A13': 25.793}

We see that the transformer correctly selected the features that have an IV score greater than :code:`threshold`
which was set to 0.2.

With :code:`transform()` we can go ahead and drop the features that do not meet the threshold:

.. code:: python

    Xtr = sel.transform(X_test)

    Xtr.head()

.. code:: python

            A2     A3 A4 A5 A7      A8 A9  A11 A13    A14  A15
    564  42.17   5.04  u  g  h  12.750  t    0   g   92.0    0
    519  39.17   1.71  u  g  v   0.125  t    5   g  480.0    0
    14   45.83  10.50  u  g  v   5.000  t    7   g    0.0    0
    257  20.00   0.00  u  g  v   0.500  f    0   g  144.0    0
    88   34.00   4.50  u  g  v   1.000  t    0   g  240.0    0


Note that :code:`Xtr` includes numerical features - A2, A3, A8, A11, and A14 - because :class:`SelectByInformationValue()`
only filters out categorical features. Also, features A4 and A5 remain because these variables were not passed to transformer
when it was instantiated.

And, finally, we can also obtain the names of the features in the final transformed dataset:

.. code:: python

    sel.get_feature_name_names_out()

    ['A2', 'A3', 'A4', 'A5', 'A7', 'A8', 'A9', 'A11', 'A13', 'A14', 'A15']