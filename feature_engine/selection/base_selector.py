import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)


def get_feature_importances(estimator):
    """Retrieve feature importances from a fitted estimator"""

    importances = getattr(estimator, "feature_importances_", None)

    coef_ = getattr(estimator, "coef_", None)

    if coef_ is not None:

        if estimator.coef_.ndim == 1:
            importances = np.abs(coef_)

        else:
            importances = np.linalg.norm(coef_, axis=0, ord=len(estimator.coef_))

    return list(importances)


class BaseSelector(BaseEstimator, TransformerMixin):
    """Transformation shared by all selectors"""

    def transform(self, X: pd.DataFrame):
        """
        Return dataframe with selected features.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features].
            The input dataframe.

        Returns
        -------
        X_transformed: pandas dataframe of shape = [n_samples, n_selected_features]
            Pandas dataframe with the selected features.
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.input_shape_[1])

        # return the dataframe with the selected features
        return X.drop(columns=self.features_to_drop_)
