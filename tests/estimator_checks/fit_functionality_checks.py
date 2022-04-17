"""Checks functionality in the fit method shared by all transformers."""

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
