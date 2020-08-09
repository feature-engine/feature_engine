
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import _define_variables


class DropFeatures(BaseEstimator, TransformerMixin):
    """
    The FeatureEliminator() drops the list of variable(s) as provided by the user
    from the dataframe and returns the subset of original dataframe with remaining
    variables.

    Parameters
    ----------

    features_to_drop : str or list, default=None
        Desired variable/s to be dropped from the dataframe

    """

    def __init__(self, features_to_drop=None):
        self.features = _define_variables(features_to_drop)

    def fit(self, X, y=None):
        """
        Verifies that the passed input X if of the type pandas dataframe

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe on which the feature elimination has to be performed

        y: None
            y is not needed for this transformer

        """
        # check input dataframe
        X = _is_dataframe(X)

        # add input shape
        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Drops the variable or list of variables provided from the original dataframe
        and returns the dataframe with subset of variables.

        Parameters
        ----------
        X: pandas dataframe
            The input dataframe on which the feature elimination has to be performed

        Returns
        -------
        X_transformed: pandas dataframe of shape = [n_samples, n_features - len(features_to_drop)]
            The transformed dataframe with subset of variables.

        """
        # check if fit is called prior
        check_is_fitted(self)

        # check input dataframe
        X = _is_dataframe(X)

        # check for input consistency
        _check_input_matches_training_df(X, self.input_shape_[1])

        X = X.copy()
        X = X.drop(columns=self.features)

        # check for a case where all columns are dropped
        if X.shape[1] == 0:
            warnings.warn(
                "The resulting dataframe has no columns after dropping all existing variables"
            )

        return X


