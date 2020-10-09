from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)
from feature_engine.variable_manipulation import _find_all_variables, _define_variables


class DropDuplicateFeatures(BaseEstimator, TransformerMixin):
    """
    DropDuplicateFeatures finds and removes duplicated features in a dataframe.

    Duplicated features are identical features, regardless of the variable or column
    name. If they show the same values for every observation, then they are considered
    duplicated.

    The transformer will first identify and store the duplicated variables. Next, the
    transformer will drop these variables from a dataframe.

    Parameters
    ----------

    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset.
    """

    def __init__(self, variables=None):
        self.variables = _define_variables(variables)

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

        duplicated_features_: set
            The duplicated features.

        duplicated_feature_sets_: list
            Groups of duplicated features. Or in other words, features that are
            duplicated with each other. Each list represents a group of duplicated
            features.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are in the dataframe
        self.variables = _find_all_variables(X, self.variables)

        # create tuples of duplicated feature groups
        self.duplicated_feature_sets_ = []

        # set to collect features that are duplicated
        self.duplicated_features_ = set()

        # create set of examined features
        _examined_features = set()

        for feature in self.variables:

            # append so we can remove when we create the combinations
            _examined_features.add(feature)

            if feature not in self.duplicated_features_:

                _temp_set = set([feature])

                # features that have not been examined, are not currently examined and
                # were not found duplicates
                _features_to_compare = [
                    f
                    for f in self.variables
                    if f not in _examined_features.union(self.duplicated_features_)
                ]

                # create combinations:
                for f2 in _features_to_compare:

                    if X[feature].equals(X[f2]):
                        self.duplicated_features_.add(f2)
                        _temp_set.add(f2)

                # if there are duplicated features
                if len(_temp_set) > 1:
                    self.duplicated_feature_sets_.append(_temp_set)

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Drops the duplicated features from a dataframe.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input samples.

        Returns
        -------
        X_transformed: pandas dataframe,
            shape = [n_samples, n_features - (duplicated features)]
            The transformed dataframe with the remaining subset of variables.

        """
        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.input_shape_[1])

        # returned non-duplicate features
        X = X.drop(columns=self.duplicated_features_)

        return X
