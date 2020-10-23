import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import _is_dataframe, \
    _check_input_matches_training_df

from feature_engine.variable_manipulation import _define_variables, \
    _find_all_variables


class ShuffleFeatures(BaseEstimator, TransformerMixin):

    """
    
    ShuffleFeatures reorganizes the values inside each feature from a dataframe.
    

    Parameters
    ----------

    features_to_shuffle : str or list, default=None
        The list of variable(s) to be shuffled from the dataframe
        If None, the transformer will shuffle all variables in the dataset.
    """

    def __init__(self, features_to_shuffle=None):

        self.features_to_shuffle = \
            _define_variables(features_to_shuffle)

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        
        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe

        y: None
            y is not needed for this transformer. You can pass y or None.


        Attributes
        ----------

        shuffled_features_: dict
            The shuffled features values

        """

        # check input dataframe

        X = _is_dataframe(X)

        # find all variables or check those entered are in the dataframe

        self.features_to_shuffle = _find_all_variables(X,
                self.features_to_shuffle)

        # dict to collect features that are shuffled

        self.shuffled_features_ = {}

        # shuffle features and save into a dict

        for feature in self.features_to_shuffle:

            shuffled_feature = random.sample(list(X[feature]), len(X))

            self.self.shuffled_features_[feature] = shuffled_feature

        return self

    def transform(self, X):
        """
        
        Updates the X dataframe with the new shuffled features.
    
        Parameters
        ----------
        
        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe from which feature values will be shuffled.


        Returns
        -------

        X_transformed: pandas dataframe of shape = [n_samples, n_features]
            Pandas dataframe with shuffled features
        """

        # check if fit is performed prior to transform

        check_is_fitted(self)

        # check if input is a dataframe

        X = _is_dataframe(X)

        # Overwrite old features with shuffled ones

        X = X.assign(**self.shuffled_features_)

        return X
