import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.validation import _return_tags


class BaseImputer(BaseEstimator, TransformerMixin):
    """shared set-up checks and methods across imputers"""

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Check that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X: Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            If the dataframe is not of same size as that used in fit()

        Returns
        -------
        X: Pandas DataFrame
            The same dataframe entered by the user.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input df contains same number of columns as df used to fit
        _check_input_matches_training_df(X, self.n_features_in_)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing data with the learned parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            If the dataframe has different number of features than the df used in fit()

        Returns
        -------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the selected variables.
        """

        X = self._check_transform_input_and_state(X)

        # replaces missing data with the learned parameters
        for variable in self.imputer_dict_:
            X[variable].fillna(self.imputer_dict_[variable], inplace=True)

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict
