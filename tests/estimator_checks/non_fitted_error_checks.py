"""Checks functionality in the transform method shared by all transformers."""

import pytest
from sklearn import clone
from sklearn.exceptions import NotFittedError

from tests.estimator_checks.dataframe_for_checks import test_df


def check_raises_non_fitted_error(estimator):
    """
    Check if transformer raises error when transform() method is called before
    calling fit() method.

    The functionality is provided by sklearn's `check_is_fitted` function.
    """
    X, y = test_df()
    transformer = clone(estimator)
    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        transformer.transform(X)
