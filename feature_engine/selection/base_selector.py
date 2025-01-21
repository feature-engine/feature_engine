import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine.dataframe_checks import _check_X_matches_training_df, check_X
from feature_engine.tags import _return_tags


class BaseSelector(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
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

    def _check_variable_number(self) -> None:
        """Check that there are multiple variables for the selectors to work with."""
        if len(self.variables_) < 2:
            raise ValueError(
                "The selector needs at least 2 or more variables to select from. "
                f"Got only 1 variable: {self.variables_}."
            )

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True if its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        mask = [
            True if f not in self.features_to_drop_ else False
            for f in self.feature_names_in_
        ]
        return mask if not indices else np.where(mask)[0]

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"

        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
