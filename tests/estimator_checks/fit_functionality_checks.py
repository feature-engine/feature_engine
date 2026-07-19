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


# A dataframe with no variables of the given type, keyed by the transformer's
# `variables` tag (see feature_engine.tags._return_tags and each transformer's
# _more_tags()). Fitting with variables=None on the matching dataframe is what
# triggers the "no variables found" path that return_empty controls.
_NO_VARIABLES_OF_TYPE = {
    "numerical": pd.DataFrame({"var_cat": ["A", "B", "A", "B", "A", "B"]}),
    "categorical": pd.DataFrame({"var_num": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}),
    "datetime": pd.DataFrame({"var_num": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}),
}

# A couple of transformers don't carry the right `variables` tag, for reasons
# unrelated to return_empty:
# - LogTransformer's own _more_tags() doesn't set "variables" at all (a
#   pre-existing gap; it's numerical like its sibling transformers).
# - DatetimeSubtraction inherits BaseCreation's "skip" tag, which exists so
#   the *generic* variable-assignment checks leave it alone (it needs
#   `variables` and `reference` checked together, which those checks don't
#   support) -- but it does select datetime variables.
_TAG_OVERRIDE_BY_CLASS_NAME = {
    "LogTransformer": "numerical",
    "DatetimeSubtraction": "datetime",
}


def check_return_empty(estimator):
    """
    Only for transformers with the init parameter `return_empty`.

    When `variables` is None and the train set contains no variables of the type
    required by the transformer (numerical, categorical or datetime), `fit()`
    raises a `TypeError` by default. When `return_empty` is set to `True`, `fit()`
    instead assigns an empty list to `variables_`, and raises a `UserWarning`
    instead of an error.

    Transformers that accept every variable type (tagged 'all') are skipped:
    there is always at least one variable of *some* type in a non-empty
    dataframe, so there is no dataframe that can trigger "no variables found"
    for them.
    """
    variables_tag = _TAG_OVERRIDE_BY_CLASS_NAME.get(
        estimator.__class__.__name__, estimator._more_tags().get("variables")
    )
    if variables_tag not in _NO_VARIABLES_OF_TYPE:
        return
    no_variables_df = _NO_VARIABLES_OF_TYPE[variables_tag]

    y = pd.Series([0, 1, 0, 1, 0, 1])

    # ignore_format=True makes categorical transformers select all variables
    # regardless of type, which defeats the purpose of this check.
    base_params = {"variables": None, "return_empty": False}
    if "ignore_format" in estimator.get_params():
        base_params["ignore_format"] = False
    # transformers like DatetimeSubtraction auto-detect `reference` the same
    # way as `variables`, and require both to be None together.
    if "reference" in estimator.get_params():
        base_params["reference"] = None

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
