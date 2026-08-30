"""Adapter that lets sklearn's ``check_estimator`` run against Feature-engine
transformers.

``check_estimator`` feeds transformers numpy arrays. Feature-engine transformers
only accept dataframes: ``feature_engine.dataframe_checks.check_X`` raises
``TypeError`` on anything that is not a dataframe from a narwhals-supported
library. Historically ``check_X`` itself converted numpy arrays into pandas
dataframes (naming the columns ``x0``, ``x1``, ...) precisely so that these
tests could run; that conversion is being removed as the checks migrate to
narwhals.

``wrap_for_check_estimator`` moves that adaptation into the test layer. It wraps
a transformer in a thin meta-estimator that turns the numpy arrays coming from
``check_estimator`` into pandas dataframes -- with the same ``x{i}`` column names
``check_X`` used to assign -- before delegating to the wrapped transformer. Tags
are delegated too, so ``check_estimator`` runs exactly the same set of checks it
would run against the bare transformer, and the ``expected_failed_checks``
dictionaries in the test files stay unchanged.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted


def wrap_for_check_estimator(estimator):
    """Wrap ``estimator`` so sklearn's ``check_estimator`` can feed it arrays.

    Parameters
    ----------
    estimator : feature-engine transformer instance.

    Returns
    -------
    _SklearnCheckInputWrapper
        A meta-estimator delegating to ``estimator`` that converts numpy array
        input into a pandas dataframe before every call.
    """
    return _SklearnCheckInputWrapper(estimator)


class _SklearnCheckInputWrapper(TransformerMixin, BaseEstimator):
    """Test-only shim: feeds the wrapped transformer a pandas dataframe when
    sklearn's ``check_estimator`` passes numpy arrays.

    Reproduces the numpy-array handling that
    ``feature_engine.dataframe_checks.check_X`` used to provide.
    """

    def __init__(self, estimator=None, random_state=None):
        self.estimator = estimator
        # Surfaced as a top-level param so sklearn's ``set_random_state`` (which
        # only looks at shallow param names) can seed the wrapped estimator in
        # the checks that require deterministic output, e.g.
        # ``check_estimators_pickle`` / ``check_pipeline_consistency``.
        self.random_state = random_state

    @staticmethod
    def _to_df(X):
        """Convert a numpy array into a dataframe, mirroring the old ``check_X``.

        Dataframes are returned unchanged. Scalars, 1D arrays and complex data
        raise the same errors ``check_X`` used to raise for them.
        """
        if hasattr(X, "iloc"):
            return X

        arr = np.asarray(X)
        if arr.ndim == 0:
            raise ValueError(
                "Expected 2D array, got scalar array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if your "
                "data has a single feature or array.reshape(1, -1) if it "
                "contains a single sample.".format(arr)
            )
        if arr.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if your "
                "data has a single feature or array.reshape(1, -1) if it "
                "contains a single sample.".format(arr)
            )
        if np.iscomplexobj(arr):
            raise TypeError("Complex data not supported by this transformer.")

        return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])

    def fit(self, X, y=None):
        self.estimator_ = clone(self.estimator)
        if (
            self.random_state is not None
            and "random_state" in self.estimator_.get_params()
        ):
            self.estimator_.set_params(random_state=self.random_state)
        if y is None:
            self.estimator_.fit(self._to_df(X))
        else:
            self.estimator_.fit(self._to_df(X), y)

        self.n_features_in_ = self.estimator_.n_features_in_
        for attr in ("feature_names_in_", "classes_"):
            if hasattr(self.estimator_, attr):
                setattr(self, attr, getattr(self.estimator_, attr))
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.estimator_.transform(self._to_df(X))

    def inverse_transform(self, X):
        check_is_fitted(self)
        return self.estimator_.inverse_transform(self._to_df(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.estimator_.get_feature_names_out(input_features)

    def __sklearn_tags__(self):
        return self.estimator.__sklearn_tags__()

    def _more_tags(self):
        return self.estimator._more_tags()
