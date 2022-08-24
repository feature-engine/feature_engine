from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._check_input_parameters.check_init_input_params import (
    _check_param_drop_original,
    _check_param_missing_values,
)
from feature_engine._variable_handling.variable_type_selection import (
    _find_or_check_numerical_variables,
)
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.tags import _return_tags


class BaseCreation(BaseEstimator, TransformerMixin):
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
        Common set-up of creation transformers.

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
        self.variables: List[Union[str, int]] = _find_or_check_numerical_variables(
            X, self.variables
        )

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables)
            _check_contains_inf(X, self.variables)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
            _check_contains_na(X, self.variables)
            _check_contains_inf(X, self.variables)

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


class GetFeatureNamesOutMixin:
    def _check_input_features(self, input_features):

        check_is_fitted(self)

        msg = f"""input features must be None or a list with one or more of the variables
        that were used by this transformer. Got {input_features} instead."""

        if input_features is None:
            # Return original variables
            input_features_ = self.variables_  # type: ignore
        else:
            if not isinstance(input_features, list):
                raise ValueError(msg)
            if any([
                f for f in input_features
                if f not in self.variables_  # type: ignore
            ]):
                raise ValueError(msg)
            # Return only features entered by user.
            input_features_ = input_features

        return input_features_

    def _return_feature_names(self, input_features, feature_names, drop_features):
        # Return names of all variables if input_features is None.
        if input_features is None or input_features is False:
            if self.drop_original is True:  # type: ignore
                # Remove names of variables to drop.
                original = [
                    f for f in self.feature_names_in_  # type: ignore
                    if f not in drop_features
                ]
                feature_names = original + feature_names
            else:
                feature_names = self.feature_names_in_ + feature_names  # type: ignore

        return feature_names
