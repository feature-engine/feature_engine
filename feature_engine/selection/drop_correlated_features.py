from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)
from feature_engine.variable_manipulation import (
    _find_numerical_variables,
    _define_variables,
)


class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """
    DropCorrelatedFeatures finds and removes correlated features.

    Features are removed on first found first removed basis, without any further
    insight.

    DropCorrelatedFeatures() works only with numerical variables. Categorical variables
    will need to be encoded to numerical or will be excluded from the analysis.

    Parameters
    ----------

    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        numerical variables in the dataset.

    method: string, default='pearson'
        Can take 'pearson', 'spearman' or'kendall'. It refers to the correlation method
        to be used to identify the correlated features.

        pearson : standard correlation coefficient
        kendall : Kendall Tau correlation coefficient
        spearman : Spearman rank correlation

        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
        for more details.

    threshold: float, default=0.8
        The correlation threshold above which a feature will be deemed correlated with
        another one and removed from the dataset.

    Attributes
    ----------

    correlated_features_: set
        The correlated features.

    correlated_feature_sets_: list
        Groups of correlated features. Or in other words, features that are
        correlated with each other. Each list represents a group of correlated
        features.

    correlated_matrix_: pandas dataframe
        The correlated matrix.

    Methods
    -------

     fit: finds the correlated features

     transform: removes correlated features

     fit_transform: finds and removes correlated features
    """

    def __init__(self, variables=None, method="pearson", threshold=0.8):

        if method not in ["pearson", "spearman", "kendall"]:
            raise ValueError(
                "correlation method takes only values 'pearson', 'spearman', 'kendall'"
            )

        if (threshold < 0 or threshold > 1) or not isinstance(threshold, float):
            raise ValueError("threshold must be a float between 0 and 1")

        self.variables = _define_variables(variables)
        self.method = method
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Finds the correlated features

        Args:
            X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

            y: It is not needed in this transformer. Defaults to None.
            Alternatively takes Pandas Series.ss

        Returns:
            self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find all numerical variables or check those entered are in the dataframe
        self.variables = _find_numerical_variables(X, self.variables)

        # set to collect features that are correlated
        self.correlated_features_ = set()

        # create tuples of correlated feature groups
        self.correlated_feature_sets_ = []

        # the correlation matrix
        self.correlated_matrix_ = X[self.variables].corr(method=self.method)

        # create set of examined features, helps to determine feature combinations
        # to evaluate below
        _examined_features = set()

        # for each feature in the dataset (columns of the correlation matrix)
        for feature in self.correlated_matrix_.columns:

            if feature not in _examined_features:

                # append so we can exclude when we create the combinations
                _examined_features.add(feature)

                # here we collect potentially correlated features
                # we need this for the correlated groups sets
                _temp_set = set([feature])

                # features that have not been examined, are not currently examined and
                # were not found correlated
                _features_to_compare = [
                    f
                    for f in self.correlated_matrix_.columns
                    if f not in _examined_features
                ]

                # create combinations:
                for f2 in _features_to_compare:

                    # if the correlation is higher than the threshold
                    # we are interested in absolute correlation coefficient value
                    if abs(self.correlated_matrix_.loc[f2, feature]) > self.threshold:

                        # add feature (f2) to our correlated set
                        self.correlated_features_.add(f2)
                        _temp_set.add(f2)
                        _examined_features.add(f2)

                # if there are correlated features
                if len(_temp_set) > 1:
                    self.correlated_feature_sets_.append(_temp_set)

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Drops the correlated features from a dataframe.

        Args:
            X: pandas dataframe of shape = [n_samples, n_features].
            The input samples.

        Returns:
            X_transformed: pandas dataframe
            shape = [n_samples, n_features - (correlated features)]
            The transformed dataframe with the remaining subset of variables.
        """
        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.correlated_matrix_.count())

        # returned non-duplicate features
        X = X.drop(columns=self.correlated_features_)

        return X
