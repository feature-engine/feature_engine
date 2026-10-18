import numpy as np
import pytest
from scipy import sparse

from tests.estimator_checks.sklearn_check_wrapper import _SklearnCheckInputWrapper


@pytest.mark.parametrize("format", ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"])
@pytest.mark.parametrize("kind", ["matrix", "array"])
def test_sparse_input_raises_explicit_error(format, kind):
    constructor = getattr(sparse, f"{format}_{kind}", None)
    if constructor is None:
        pytest.skip("Sparse array format unavailable in this SciPy version.")
    X = constructor(np.eye(3))

    with pytest.raises(TypeError, match="[Ss]parse.*not supported"):
        _SklearnCheckInputWrapper._to_df(X)
