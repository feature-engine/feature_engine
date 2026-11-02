"""Regression coverage for sparse inputs to the sklearn test-only adapter."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from feature_engine.selection import DropCorrelatedFeatures
from tests.estimator_checks.sklearn_check_wrapper import (
    _SklearnCheckInputWrapper,
    wrap_for_check_estimator,
)


@pytest.mark.parametrize(
    "factory", [sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix]
)
def test_to_df_rejects_sparse_input_with_explicit_message(factory):
    X = factory([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError, match="(?i)sparse"):
        _SklearnCheckInputWrapper._to_df(X)


def test_to_df_keeps_dense_array_conversion():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _SklearnCheckInputWrapper._to_df(X)
    pd.testing.assert_frame_equal(result, pd.DataFrame(X, columns=["x0", "x1"]))


def test_wrapped_estimator_fit_rejects_sparse_with_explicit_message():
    wrapped = wrap_for_check_estimator(DropCorrelatedFeatures())
    with pytest.raises(TypeError, match="(?i)sparse"):
        wrapped.fit(sparse.csr_matrix(np.eye(3)))
