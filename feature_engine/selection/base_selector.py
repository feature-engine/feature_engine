import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.variable_manipulation import _filter_out_variables_not_in_dataframe
from feature_engine.validation import _return_tags


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
        Remove non selected features.

    _confirm_variables:
        Check that the variables entered by the user exist in the df.
    """

    def __init__(
            self,
            confirm_variables: bool = False,
    ) -> None:

        if not isinstance(confirm_variables, bool):
            raise ValueError("confirm_variables takes only values True and False. "
                             f"Got {confirm_variables} instead.")

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
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.n_features_in_)

        # return the dataframe with the selected features
        return X.drop(columns=self.features_to_drop_)

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict
