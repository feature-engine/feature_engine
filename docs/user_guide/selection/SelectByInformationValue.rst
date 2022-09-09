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
:class: `SelectByInformationValue()` automatically identifies categorical variables, i.e., variable types that
are object or categorical. If any dataset's categorical variables uses numeric values for its categories or bins
the init parameter :code:`ignore_format` should be set to :code:`False`.


Example
-------
Let's see how to use this transformer to select variables from UC Irvine's credit approval data set which can
be found `here`_. This dataset concerns credit card applications. All attributes names and values have been changed
to meaningless symbols to protect confidentiality.

The data is comprised of both numeric and categorical data.

.. _here: https://archive-beta.ics.uci.edu/ml/datasets/credit+approval

Let's import the required libraries and classes.

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from feature_engine import SelectByInformationValue

Let's now load and prepare the credit approval data.

.. code:: python

    # load data
    data = pd.read_csv("data/crx.data", header=None)

    # name variables
    var_names = ["A" + str(s) for s in range(1,17)]
    data.columns = var_names
    data.rename(columns={"A16": "target"}, inplace=True)

    # preprocess data
    data = data.replace("?", np.nan)
    data["A2"] = data["A2"].astype("float")
    data["A14"] = data["A14"].astype("float")
    data["target"] = data["target"].map({"+":1, "-":0})

    data.head()

Let's now review the first 5 rows of the dataset.

.. code:: python

  A1     A2     A3 A4 A5 A6 A7    A8 A9 A10  A11 A12 A13    A14  A15  target
0  b  30.83  0.000  u  g  w  v  1.25  t   t    1   f   g  202.0    0       1
1  a  58.67  4.460  u  g  q  h  3.04  t   t    6   f   g   43.0  560       1
2  a  24.50  0.500  u  g  q  h  1.50  t   f    0   f   g  280.0  824       1
3  b  27.83  1.540  u  g  w  v  3.75  t   t    5   t   g  100.0    3       1
4  b  20.17  5.625  u  g  w  v  1.71  t   f    0   f   s  120.0    0       1