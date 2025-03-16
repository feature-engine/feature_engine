import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine.dataframe_checks import _check_X_matches_training_df, check_X
from feature_engine.tags import _return_tags


class BaseImputer(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """shared set-up checks and methods across imputers"""

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Common checks before transforming data:

        - Check transformer was fit
        - Check that the input is a dataframe
        - Check that input has same size than the train set used in fit()
        - Re-orders dataframe features if necessary

        Parameters
        ----------
        X: Pandas DataFrame

        Returns
        -------
        X: Pandas DataFrame
            The same dataframe entered by the user.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check that input df contains same number of columns as df used to fit
        _check_X_matches_training_df(X, self.n_features_in_)

        # reorder df to match train set
        X = X[self.feature_names_in_]

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing data with the learned parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the selected variables.
        """

        X = self._transform(X)

        # Replace missing data with learned parameters
        with pd.option_context("future.no_silent_downcasting", True):
            X = X.fillna(value=self.imputer_dict_).infer_objects(copy=False)
        return X

    def _get_feature_names_in(self, X):
        """Get the names and number of features in the train set (the dataframe
        used during fit)."""

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "numerical"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
