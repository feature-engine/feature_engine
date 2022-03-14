.. _relative_features:

.. currentmodule:: feature_engine.creation

CombineWithReferenceFeature
===========================

:class:`CombineWithReferenceFeature()` combines a group of variables with a group of
reference variables utilizing basic mathematical operations (subtraction, division,
addition and multiplication). It returns one or more additional features in the
dataframe as a result of these operations.

In other words, :class:`CombineWithReferenceFeature()` sums, multiplies, subtracts or
divides a group of features (indicated in `variables_to_combine`) to or by a group of
reference variables (indicated in `reference_variables`), and returns the
result as new variables in the dataframe.

For example, if we have the variables:

- **number_payments_first_quarter**,
- **number_payments_second_quarter**,
- **number_payments_third_quarter**,
- **number_payments_fourth_quarter**, and
- **total_payments**,

we can use :class:`CombineWithReferenceFeature()` to determine the percentage of
payments per quarter as follows:

.. code-block:: python

    transformer = CombineWithReferenceFeature(
        variables_to_combine=[
            'number_payments_first_quarter',
            'number_payments_second_quarter',
            'number_payments_third_quarter',
            'number_payments_fourth_quarter',
        ],

        reference_variables=['total_payments'],

        operations=['div'],

        new_variables_name=[
            'perc_payments_first_quarter',
            'perc_payments_second_quarter',
            'perc_payments_third_quarter',
            'perc_payments_fourth_quarter',
        ]
    )

    Xt = transformer.fit_transform(X)

The precedent code block will return a new dataframe, Xt, with 4 new variables, those
indicated in `new_variables_name`, that are calculated as the division of each one of
the variables in `variables_to_combine` and 'total_payments'.

Below we show another example using the House Prices Dataset (more details about the
dataset :ref:`here <datasets>`). In this example, we subtract `LotFrontage` from
`LotArea`.

.. code:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.creation import CombineWithReferenceFeature

    data = pd.read_csv('houseprice.csv').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0
    )

    combinator = CombineWithReferenceFeature(
        variables_to_combine=['LotArea'],
        reference_variables=['LotFrontage'],
        operations = ['sub'],
        new_variables_names = ['LotPartial']
        )

    combinator.fit(X_train, y_train)

    X_train = combinator.transform(X_train)

We can see the newly created variable in the following code blocks:

.. code:: python

    print(X_train[["LotPartial","LotFrontage","LotArea"]].head())

.. code:: python

        LotTotal  LotFrontage  LotArea
    64      9375.0          0.0     9375
    682     2887.0          0.0     2887
    960     7157.0         50.0     7207
    1384    9000.0         60.0     9060
    1100    8340.0         60.0     8400

**new_variables_names**

Even though the transfomer allows to combine variables automatically, it was originally
designed to combine variables with domain knowledge. In this case, we normally want to
give meaningful names to the variables. We can do so through the parameter
`new_variables_names`.

`new_variables_names` takes a list of strings, with the new variable names. In this
parameter, you need to enter as many names as new features are created by the
transformer. The number of new features is the number of operations, times the number
of reference variables, times the number of variables to combine.

Thus, if you want to perform 2 operations, sub and div, combining 4 variables
with 2 reference variables, you should enter 2 X 4 X 2 new variable names.

The name of the variables should coincide with the order in which the operations are
performed by the transformer. The transformer will first carry out 'sub', then 'div',
then 'add' and finally 'mul'.

More details
^^^^^^^^^^^^

You can find creative ways to use the :class:`CombineWithReferenceFeature()` in the
following Jupyter notebooks and Kaggle kernels.

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/creation/CombineWithReferenceFeature.ipynb>`_
- `Kaggle kernel - Wine Quality <https://www.kaggle.com/solegalli/create-new-features-with-feature-engine>`_
- `Kaggle kernel - House Price <https://www.kaggle.com/solegalli/feature-engineering-and-model-stacking>`_
