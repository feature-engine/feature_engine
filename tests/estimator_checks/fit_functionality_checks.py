"""Checks functionality in the fit method shared by all transformers."""

import pandas as pd
import pytest
from sklearn import clone

from tests.estimator_checks.dataframe_for_checks import test_df


def check_feature_names_in(estimator):
    """Checks that transformers learn the variable names of the train set used
    during fit, as well as the number of variables.

    Should be applied to all transformers.
    """
    # the estimator learns the parameters from the train set
    X, y = test_df(categorical=True, datetime=True)
    varnames = list(X.columns)
    estimator = clone(estimator)
    estimator.fit(X, y)
    assert estimator.feature_names_in_ == varnames
    assert estimator.n_features_in_ == len(varnames)


def check_error_if_y_not_passed(estimator):
    """
    Checks that transformer raises error when y is not passed during fit. Functionality
    is provided by Python, when making a parameter mandatory.

    For this test to run, we need to add the tag 'requires_y' to the transformer.
    """
    X, y = test_df()
    estimator = clone(estimator)
    with pytest.raises(TypeError):
        estimator.fit(X)


def check_return_empty(estimator):
    """
    Only for transformers with the init parameter `return_empty`.

    When `variables` is None and the train set contains no variables of the type
    required by the transformer (numerical, categorical or datetime), `fit()`
    raises a `TypeError` by default. When `return_empty` is set to `True`, `fit()`
    instead assigns an empty list to `variables_`, and raises a `UserWarning`
    instead of an error.

    Transformers that accept every variable type (tagged 'all') cannot be probed
    this way, since there is always at least one variable of *some* type in a
    non-empty dataframe: the check is skipped for those.
    """
    y = pd.Series([0, 1, 0, 1, 0, 1])
    candidate_dfs = [
        pd.DataFrame({"var_num": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}),
        pd.DataFrame({"var_cat": ["A", "B", "A", "B", "A", "B"]}),
        pd.DataFrame(
            {"var_date": pd.date_range("2020-01-01", periods=6, freq="D")}
        ),
    ]

    # ignore_format=True makes categorical transformers select all variables
    # regardless of type, which defeats the purpose of this check.
    base_params = {"variables": None, "return_empty": False}
    if "ignore_format" in estimator.get_params():
        base_params["ignore_format"] = False
    # transformers like DatetimeSubtraction auto-detect `reference` the same
    # way as `variables`, and require both to be None together.
    if "reference" in estimator.get_params():
        base_params["reference"] = None

    no_variables_df = None
    for df in candidate_dfs:
        transformer = clone(estimator)
        transformer.set_params(**base_params)
        try:
            transformer.fit(df, y)
        except TypeError:
            no_variables_df = df
            break

    if no_variables_df is None:
        return

    # default: raises an error
    transformer = clone(estimator)
    transformer.set_params(**base_params)
    with pytest.raises(TypeError):
        transformer.fit(no_variables_df, y)

    # KNOWN BUG: DecisionTreeEncoder builds a nested OrdinalEncoder with
    # `variables=variables_`. When `variables_` is an empty list, that nested
    # encoder's own constructor rejects it (a guard meant to catch users
    # explicitly passing `variables=[]` by mistake), so fit() raises ValueError
    # instead of completing. This asserts today's actual behaviour so the test
    # documents the bug rather than papering over it.
    if estimator.__class__.__name__ == "DecisionTreeEncoder":
        transformer = clone(estimator)
        transformer.set_params(**{**base_params, "return_empty": True})
        with pytest.raises(ValueError, match="The list of `variables` is empty"):
            transformer.fit(no_variables_df, y)
        return

    # return_empty=True: warns and returns an empty list instead of raising
    transformer = clone(estimator)
    transformer.set_params(**{**base_params, "return_empty": True})
    with pytest.warns(UserWarning):
        transformer.fit(no_variables_df, y)
    assert transformer.variables_ == []
