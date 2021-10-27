.. _math_combination:

.. currentmodule:: feature_engine.creation

MathematicalCombination
=======================

:class:`MathematicalCombination()` applies basic mathematical operations to multiple
features, returning one or more additional features as a result. That is, it sums,
multiplies, takes the average, maximum, minimum or standard deviation of a group
of variables and returns the result into new variables.

For example, if we have the variables:

- **number_payments_first_quarter**,
- **number_payments_second_quarter**,
- **number_payments_third_quarter** and
- **number_payments_fourth_quarter**,

we can use :class:`MathematicalCombination()` to calculate the total number of payments
and mean number of payments as follows:

.. code-block:: python

    transformer = MathematicalCombination(
        variables_to_combine=[
            'number_payments_first_quarter',
            'number_payments_second_quarter',
            'number_payments_third_quarter',
            'number_payments_fourth_quarter'
        ],
        math_operations=[
            'sum',
            'mean'
        ],
        new_variables_name=[
            'total_number_payments',
            'mean_number_payments'
        ]
    )

    Xt = transformer.fit_transform(X)


The transformed dataset, Xt, will contain the additional features
**total_number_payments** and **mean_number_payments**, plus the original set of
variables. The variable **total_number_payments** is obtained by adding up the features
indicated in `variables_to_combine`, whereas the variable **mean_number_payments** is
the mean of those 4 features.

Below we show another example using the House Prices Dataset (more details about the
dataset :ref:`here <datasets>`). In this example, we sum 2 variables: 'LotFrontage' and
'LotArea' to obtain 'LotTotal'.

.. code:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.creation import MathematicalCombination

    data = pd.read_csv('houseprice.csv').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['Id', 'SalePrice'], axis=1),
        data['SalePrice'],
        test_size=0.3,
        random_state=0
    )

    math_combinator = MathematicalCombination(
        variables_to_combine=['LotFrontage', 'LotArea'],
        math_operations = ['sum'],
        new_variables_names = ['LotTotal']
    )

    math_combinator.fit(X_train, y_train)
    X_train_ = math_combinator.transform(X_train)

.. code:: python

    print(math_combinator.combination_dict_)

.. code:: python

    {'LotTotal': 'sum'}

.. code:: python

    print(X_train_.loc[:,['LotFrontage', 'LotArea', 'LotTotal']].head())

.. code:: python

          LotFrontage  LotArea  LotTotal
    64            0.0     9375    9375.0
    682           0.0     2887    2887.0
    960          50.0     7207    7257.0
    1384         60.0     9060    9120.0
    1100         60.0     8400    8460.0

**new_variables_names**

Even though the transfomer allows to combine variables automatically, it was originally
designed to combine variables with domain knowledge. In this case, we normally want to
give meaningful names to the variables. We can do so through the parameter
`new_variables_names`.

`new_variables_names` takes a list of strings, with the new variable names. In this
parameter, you need to enter a name or a list of names for the newly created features
(recommended). You must enter one name for each mathematical transformation indicated
in the `math_operations` parameter. That is, if you want to perform mean and sum of
features, you should enter 2 new variable names. If you perform only mean of features,
enter 1 variable name. Alternatively, if you chose to perform all mathematical
transformations, enter 6 new variable names.

The name of the variables should coincide with the order in which the
mathematical operations are initialised in the transformer. That is, if you set
math_operations = ['mean', 'prod'], the first new variable name will be
assigned to the mean of the variables and the second variable name
to the product of the variables.

More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/creation/MathematicalCombination.ipynb>`_
- `Kaggle kernel - Wine Quality <https://www.kaggle.com/solegalli/create-new-features-with-feature-engine>`_
- `Kaggle kernel - House Price <https://www.kaggle.com/solegalli/feature-engineering-and-model-stacking>`_

