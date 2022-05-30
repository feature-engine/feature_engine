"""Feature-engine transformers select variables during fit or alternatively check
that the variables entered by the user are of the allowed type.

Some transformers operate only on numerical variables, some on categorical, some
on datetime, and some on all types of variables.

This scripts contains common tests for this functionality.
"""

import pytest
from sklearn import clone

from tests.estimator_checks.dataframe_for_checks import test_df


def check_numerical_variables_assignment(estimator, needs_group=False):
    """
    Checks that transformers that work only with numerical variables, correctly set
    the values for the attributes `variables` and `variables_`.

    The first attribute can take a string, an integer, a list of strings or integers or
    None.

    The second attribute can take a list of string or integers, but is assigned after
    checking that the variables are of type numeric.

    For this check to run, the transformer needs the tag 'variables' set to 'numerical'.
    """
    # toy df
    X, y = test_df(categorical=True)

    # input variables to test
    if needs_group is True:
        _input_vars_ls = [["var_1", "var_2", "var_3", "var_11"], None]
    else:
        _input_vars_ls = [
            "var_1",
            ["var_2"],
            ["var_1", "var_2", "var_3", "var_11"],
            None,
        ]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

        # before fitting
        if input_vars is not None:
            assert transformer.variables == input_vars
        else:
            assert transformer.variables is None

        # after fitting
        transformer.fit(X, y)

        if input_vars is not None:
            assert transformer.variables == input_vars

            if isinstance(input_vars, list):
                assert transformer.variables_ == input_vars
            else:
                assert transformer.variables_ == [input_vars]
        else:
            assert transformer.variables is None
            assert transformer.variables_ == ["var_" + str(i) for i in range(12)]

    # test raises error if user passes categorical variable
    transformer.set_params(variables=["var_1", "cat_var1"])
    with pytest.raises(TypeError):
        transformer.fit(X, y)


def check_categorical_variables_assignment(estimator, needs_group=False):
    """
    Checks that transformers that work only with categorical variables, correctly set
    the values for the attributes `variables` and `variables_`.

    The first attribute can take a string, an integer, a list of strings or integers or
    None.

    The second attribute can take a list of string or integers, but is assigned after
    checking that the variables are of type object or categorical.

    For this check to run, the transformer needs the tag 'variables' set to
    'categorical'.
    """
    # toy df
    X, y = test_df(categorical=True)

    # cast one variable as category
    X[["cat_var2"]] = X[["cat_var2"]].astype("category")

    # input variables to test
    if needs_group is True:
        _input_vars_ls = [["cat_var1", "cat_var2"], None]
    else:
        _input_vars_ls = ["cat_var1", ["cat_var1"], ["cat_var1", "cat_var2"], None]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

        # before fitting
        if input_vars is not None:
            assert transformer.variables == input_vars
        else:
            assert transformer.variables is None

        # fit
        transformer.fit(X, y)

        if input_vars is not None:
            assert transformer.variables == input_vars

            if isinstance(input_vars, list):
                assert transformer.variables_ == input_vars
            else:
                assert transformer.variables_ == [input_vars]
        else:
            assert transformer.variables is None
            assert transformer.variables_ == ["cat_var1", "cat_var2"]

    # test raises error if uses passes numerical variable
    transformer.set_params(variables=["var_1", "cat_var1"])
    with pytest.raises(TypeError):
        transformer.fit(X, y)


def check_all_types_variables_assignment(estimator, needs_group=False):
    """
    Checks that transformers that work with all types of variables, correctly set
    the values for the attributes `variables` and `variables_`.

    The first attribute can take a string, an integer, a list of strings or integers or
    None. The second attribute can take a list of string or integers.

    For this check to run, the transformer needs the tag 'variables' set to
    'all'.
    """

    # toy df
    X, y = test_df(categorical=True)

    # cast one variable as category
    X[["cat_var2"]] = X[["cat_var2"]].astype("category")

    # input variables to test
    if needs_group is True:
        _input_vars_ls = [
            ["var_1", "var_2", "cat_var1", "cat_var2"],
            None,
        ]
    else:
        _input_vars_ls = [
            "var_1",
            ["cat_var1"],
            ["var_1", "var_2", "cat_var1", "cat_var2"],
            None,
        ]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

        # before fitting
        if input_vars is not None:
            assert transformer.variables == input_vars
        else:
            assert transformer.variables is None

        # fit
        transformer.fit(X, y)

        if input_vars is not None:
            assert transformer.variables == input_vars

            if isinstance(input_vars, list):
                assert transformer.variables_ == input_vars
            else:
                assert transformer.variables_ == [input_vars]
        else:
            assert transformer.variables is None
            assert transformer.variables_ == list(X.columns)


def check_datetime_variables_assignment(estimator):
    """
    Checks that transformers that work with datetime variables, correctly set
    the values for the attributes `variables` and `variables_`.

    The first attribute can take a string, an integer, a list of strings or integers or
    None. The second attribute can take a list of string or integers.

    For this check to run, the transformer needs the tag 'variables' set to
    'datetime'.
    """

    # toy df
    X, y = test_df(datetime=True)

    # input variables to test
    _input_vars_ls = ["date1", ["date2"], ["date1", "date2"], None]

    # the estimator
    transformer = clone(estimator)

    for input_vars in _input_vars_ls:
        # set the different input var examples
        transformer.set_params(variables=input_vars)

        # before fitting
        if input_vars is not None:
            assert transformer.variables == input_vars
        else:
            assert transformer.variables is None

        # fit
        transformer.fit(X, y)

        if input_vars is not None:
            assert transformer.variables == input_vars

            if isinstance(input_vars, list):
                assert transformer.variables_ == input_vars
            else:
                assert transformer.variables_ == [input_vars]
        else:
            assert transformer.variables is None
            assert transformer.variables_ == ["date1", "date2"]

    # test raises error if uses passes numerical variable
    transformer.set_params(variables=["var_1", "date1"])
    with pytest.raises(TypeError):
        transformer.fit(X, y)
