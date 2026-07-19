from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_drop_original,
    _check_param_missing_values,
)
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


class BaseCreation(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """Shared set-up, checks and methods across creation transformers."""

    def __init__(
        self,
        missing_values: str = "raise",
        drop_original: bool = False,
    ) -> None:

        _check_param_missing_values(missing_values)
        _check_param_drop_original(drop_original)

        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas Series, or np.array. Defaults to None.
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = check_X(X)

        # check variables are numerical
        if self.variables is None:
            self.variables_ = find_numerical_variables(X)
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        if hasattr(self, "reference"):
            check_numerical_variables(X, self.reference)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)
            if hasattr(self, "reference"):
                _check_contains_na(X, self.reference)
                _check_contains_inf(X, self.reference)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Common input and transformer checks.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe
            The dataframe with the original variables plus the new variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)
            if hasattr(self, "reference"):
                _check_contains_na(X, self.reference)
                _check_contains_inf(X, self.reference)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "skip"
        # Tests that are OK to fail:
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"][
            "check_fit2d_1feature"
        ] = "this transformer works with datasets that contain at least 2 variables. \
        Otherwise, there is nothing to combine"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
