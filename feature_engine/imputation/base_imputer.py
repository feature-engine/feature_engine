import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)


class BaseImputer(BaseEstimator, TransformerMixin):
    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input df contains same number of columns as df used to fit
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces missing data with the learned parameters.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the selected variables.
        """

        X = self._check_transform_input_and_state(X)

        # replaces missing data with the learned parameters
        for variable in self.imputer_dict_:
            X[variable].fillna(self.imputer_dict_[variable], inplace=True)

        return X
