from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import _check_X_matches_training_df, check_X
from feature_engine.tags import _return_tags
from feature_engine.variable_manipulation import _filter_out_variables_not_in_dataframe


def get_feature_importances(estimator):
    """Retrieve feature importance from a fitted estimator"""

    importances = getattr(estimator, "feature_importances_", None)

    coef_ = getattr(estimator, "coef_", None)

    if coef_ is not None:

        if estimator.coef_.ndim == 1:
            importances = np.abs(coef_)

        else:
            importances = np.linalg.norm(coef_, axis=0, ord=len(estimator.coef_))

        importances = list(importances)

    return importances


class BaseSelector(BaseEstimator, TransformerMixin):
    """
    Shared set-up checks and methods across selectors.

    Parameters
    ----------
    confirm_variables: bool, default=False
        If set to True, variables that are not present in the input dataframe will be
        removed from the indicated list of variables. See parameter variables for more
        details.

    Methods
    -------
    transform:
        Remove non-selected features.

    _confirm_variables:
        Check that the variables entered by the user exist in the df.
    """

    _confirm_variables_docstring = """confirm_variables: bool, default=False
            If set to True, variables that are not present in the input dataframe will
            be removed from the list of variables. Only used when passing a variable
            list to the parameter `variables`. See parameter variables for more details.
            """.rstrip()

    def __init__(
        self,
        confirm_variables: bool = False,
    ) -> None:

        if not isinstance(confirm_variables, bool):
            raise ValueError(
                "confirm_variables takes only values True and False. "
                f"Got {confirm_variables} instead."
            )

        self.confirm_variables = confirm_variables

    def _confirm_variables(self, X: pd.DataFrame) -> None:
        # If required, exclude variables that are not in the input dataframe
        if self.confirm_variables:
            self.variables_ = _filter_out_variables_not_in_dataframe(X, self.variables)
        else:
            self.variables_ = self.variables

        return None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return dataframe with selected features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_selected_features]
            Pandas dataframe with the selected features.
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = check_X(X)

        # check if number of columns in test dataset matches to train dataset
        _check_X_matches_training_df(X, self.n_features_in_)

        # reorder df to match train set
        X = X[self.feature_names_in_]

        # return the dataframe with the selected features
        return X.drop(columns=self.features_to_drop_)

    def _get_feature_names_in(self, X):
        """Get the names and number of features in the train set. The dataframe
        used during fit."""

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self

    def get_feature_names_out(self, input_features=None) -> List:
        """Get output feature names for transformation.

        input_features: None
            This parameter exists only for compatibility with the Scikit-learn
            pipeline, but has no functionality. You can pass a list of feature names
            or None.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        feature_names = [
            f for f in self.feature_names_in_ if f not in self.features_to_drop_
        ]

        return feature_names

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict
