from typing import Union

import narwhals as nw
from narwhals.typing import IntoDataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine.dataframe_checks import _check_X_matches_training_df, check_X
from feature_engine.dataframe_helpers import fill_missing_in_df, subset_df
from feature_engine.tags import _return_tags


class BaseImputer(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """shared set-up checks and methods across imputers"""

    def _transform(self, X: IntoDataFrame) -> Union[IntoDataFrame, nw.DataFrame]:
        """
        Common checks before transforming data:

        - Check transformer was fit
        - Check that the input is a dataframe
        - Check that input has same size than the train set used in fit()
        - Re-orders dataframe features if necessary

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]

        Returns
        -------
        X: pandas or narwhals dataframe
            The same pandas dataframe if X is pandas, or X in narwhals format
            otherwise, with the variables in the same order as in the train set.
        """
        check_is_fitted(self)
        X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)
        return subset_df(X, self.feature_names_in_)

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Replace missing data with the learned parameters.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the selected variables.
        """
        X = self._transform(X)
        X = fill_missing_in_df(X, self.imputer_dict_)
        return nw.to_native(X, pass_through=True)

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "numerical"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
