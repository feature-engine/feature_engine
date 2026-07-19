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
    instead of an error. Transformer should return the same dataframe in this case.
    """
    # dataframe with no variables of the given type
    variable_tag = estimator._more_tags().get("variables")
    if variable_tag in ["numerical", "datetime"]:
        df = pd.DataFrame({"var_cat": ["A", "B", "A", "B", "A", "B"]})
    elif variable_tag in ["all", "skip"]:
        return
    else:
        df = pd.DataFrame({"var_num": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})

    y = pd.Series([0, 1, 0, 1, 0, 1])

    # ignore_format=True makes categorical transformers select all variables
    # regardless of type, which defeats the purpose of this check.
    base_params = {"variables": None, "return_empty": False}
    if "ignore_format" in estimator.get_params():
        base_params["ignore_format"] = False

    # default: raises an error
    transformer = clone(estimator)
    transformer.set_params(**base_params)
    with pytest.raises(TypeError):
        transformer.fit(df, y)

    # return_empty=True: warns and returns an empty list instead of raising
    transformer = clone(estimator)
    transformer.set_params(**{**base_params, "return_empty": True})
    with pytest.warns(UserWarning):
        transformer.fit(df, y)
    assert transformer.variables_ == []

    # if return_empty=True, transformer should return same df
    # after transformation
    dft = transformer.transform(df)
    pd.testing.assert_frame_equal(dft, df)
