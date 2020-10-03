import itertools
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df


class DropDuplicateFeatures(BaseEstimator, TransformerMixin):
    """
    Drop duplicate features from a dataframe. Duplicate features are those set of features which show same value across
    all observations.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):

        """
        Find duplicated features.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.

        y: None
            y is not needed for this transformer. You can pass y or None.


        Attributes
        ----------

        duplicate_feature_dict_: dict
            The dictionary where keys are features and values are duplicate features.
        """

        # check input dataframe
        X = _is_dataframe(X)

        duplicated_features_list = []

        # create column pairs
        col_pair = list(itertools.combinations(X.columns, 2))

        # create a dictionary of features with values as list of duplicate features
        self.duplicate_feature_dict_ = {col: [] for col in X.columns}

        for feat_1, feat_2 in col_pair:
            if feat_1 not in duplicated_features_list and X[feat_1].equals(X[feat_2]):
                self.duplicate_feature_dict_[feat_1].append(feat_2)
                duplicated_features_list.append(feat_2)

        # remove features which are found as duplicate
        self.duplicate_feature_dict_ = {key: value for key, value in self.duplicate_feature_dict_.items()
                                        if key not in duplicated_features_list}

        self.input_shape_ = X.shape

        return self

    def transform(self, X):

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.input_shape_[1])

        # returned non-duplicate features
        X = X[self.duplicate_feature_dict_.keys()]

        return X
