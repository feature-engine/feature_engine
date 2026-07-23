"""Checks functionality in the transform method shared by all transformers."""

import pytest
from sklearn import clone
from sklearn.exceptions import NotFittedError

from tests.estimator_checks.dataframe_for_checks import test_df


def check_raises_non_fitted_error(estimator):
    """
    Check if transformer raises error when transform(), inverse_transform(),
    get_feature_names_out() or other fit-dependent methods are called before
    calling fit() method.

    For transform(), get_feature_names_out(), and for inverse_transform() when
    the transformer implements it, the expected error is NotFittedError,
    provided by sklearn's `check_is_fitted` function.

    A few transformers expose inverse_transform() but do not implement it (for
    example, OneHotEncoder, RareLabelEncoder and StringSimilarityEncoder). Those
    raise NotImplementedError instead, whether or not fit() was called. We assert
    the specific error each transformer is expected to raise, rather than
    accepting either, so that a transformer that stops guarding fit() is caught.

    A few transformers expose extra methods that also require a fitted
    transformer, for example, transform_x_y() and return_na_data() in
    DropMissingData. Those are checked too, when present.
    """
    X, y = test_df()

    # Test when fit is not called prior to transform.
    transformer = clone(estimator)
    with pytest.raises(NotFittedError):
        transformer.transform(X)

    # Test when fit is not called prior to inverse_transform.
    if hasattr(estimator, "inverse_transform"):
        expected_error = (
            NotFittedError
            if _implements_inverse_transform(estimator)
            else NotImplementedError
        )
        transformer = clone(estimator)
        with pytest.raises(expected_error):
            transformer.inverse_transform(X)

    # Test when fit is not called prior to get_feature_names_out.
    transformer = clone(estimator)
    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out()

    # Test when fit is not called prior to transform_x_y.
    if hasattr(estimator, "transform_x_y"):
        transformer = clone(estimator)
        with pytest.raises(NotFittedError):
            transformer.transform_x_y(X, y)

    # Test when fit is not called prior to return_na_data.
    if hasattr(estimator, "return_na_data"):
        transformer = clone(estimator)
        with pytest.raises(NotFittedError):
            transformer.return_na_data(X)


def _implements_inverse_transform(estimator):
    """Whether the transformer actually implements inverse_transform.

    Transformers that expose the method but do not support the inversion raise
    NotImplementedError. We detect this on a fitted clone: if inverse_transform
    raises NotImplementedError once fitted, the method is not implemented; any
    other outcome means it is implemented (and should therefore raise
    NotFittedError when called before fit).

    Fitting uses a dataframe with categorical and datetime features so that every
    transformer that reaches this check can be fitted, matching the input used by
    the other shared checks.
    """
    X, y = test_df(categorical=True, datetime=True)
    transformer = clone(estimator)
    transformer.fit(X, y)
    try:
        transformer.inverse_transform(X)
    except NotImplementedError:
        return False
    except Exception:
        # inverse_transform is implemented but raised for an unrelated reason
        # (e.g. the raw X is not valid transformed input); it is not "not
        # implemented", which is all this check needs to establish.
        return True
    return True
