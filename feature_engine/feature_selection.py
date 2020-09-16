import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import _define_variables


class DropFeatures(BaseEstimator, TransformerMixin):
    """
    DropFeatures() drops the list of variable(s) indicated by the user
    from the original dataframe and returns the remaining variables.

    Parameters
    ----------

    features_to_drop : str or list, default=None
        Variable(s) to be dropped from the dataframe

    """

    def __init__(self, features_to_drop=None):

        self.features_to_drop = _define_variables(features_to_drop)

        if len(self.features_to_drop) == 0:
            raise ValueError('List of features to drop cannot be empty. Please pass at least 1 variable to drop')

    def fit(self, X, y=None):
        """
        Verifies that the input X is a pandas dataframe

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe

        y: None
            y is not needed for this transformer. You can pass y or None.

        """
        # check input dataframe
        X = _is_dataframe(X)

        # check for non existent columns
        non_existent = [x for x in self.features_to_drop if x not in X.columns]
        if non_existent:
            raise KeyError(
                f"Columns '{', '.join(non_existent)}' not present in the input dataframe, "
                f"please check the columns and enter a new list of features to drop"
            )

        # check that user does not drop all columns returning empty dataframe
        if len(self.features_to_drop) == len(X.columns):
            raise ValueError("The resulting dataframe will have no columns after dropping all existing variables")

        # add input shape
        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Drops the variable or list of variables indicated by the user from the original dataframe
        and returns a new dataframe with the remaining subset of variables.

        Parameters
        ----------
        X: pandas dataframe
            The input dataframe from which features will be dropped

        Returns
        -------
        X_transformed: pandas dataframe of shape = [n_samples, n_features - len(features_to_drop)]
            The transformed dataframe with the remaining subset of variables.

        """
        # check if fit is called prior
        check_is_fitted(self)

        # check input dataframe
        X = _is_dataframe(X)

        # check for input consistency
        _check_input_matches_training_df(X, self.input_shape_[1])

        X = X.drop(columns=self.features_to_drop)

        return X


class SelectFeatures(TransformerMixin, BaseEstimator):
    """
    SelectFeatures()

    Drops constant and quasi constant variables from a dataframe.
    Constant variables are those variables which show only one value for all the observations.
    By default, all constant variables are dropped.
    Quasi-constant variables are those variables which are almost constant. In other words, those features have the same
    value for large subset of total observations.
    This transformer works for both, numerical and categorical variables.

    Parameters
    ----------

    drop_quasi_constant_features : bool, default=False
        Whether to drop quasi constant features

    quasi_constant_threshold: float, default=None
        When drop_quasi_constant_features is True, features showing quasi_constant_threshold percentage of values
        will be dropped

    Attributes
    ----------

    constant_features_: list
        List of constant features in a dataframe

    quasi_constant_features_: list
        List of quasi constant features in a dataframe

    selected_features_: list
        List of selected features which are non-constant and non-quasi constant

    """
    def __init__(self, drop_quasi_constant_features=False, quasi_constant_threshold=None):

        self.drop_quasi_constant_features = drop_quasi_constant_features
        self.quasi_constant_threshold = quasi_constant_threshold

        self.constant_features_ = []
        self.selected_features_ = []
        self.quasi_constant_features_ = []
        self._input_shape_ = ()

        if not self.drop_quasi_constant_features and self.quasi_constant_threshold is not None:
            warnings.warn(
                'Setting a quasi_constant_threshold has no effect since drop_quasi_constant_features is False. You '
                'should leave quasi_constant_threshold to its default (None), or set drop_quasi_constant_features=True.'
                , FutureWarning
            )

        if self.drop_quasi_constant_features:
            if self.quasi_constant_threshold <= 0 or self.quasi_constant_threshold >= 1:
                raise ValueError("Value of threshold should be between 0 and 1")

    def fit(self, X, y=None):

        """
        Find constant and non-quasi constant features from a dataframe

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe

        y: None
            y is not needed for this transformer. You can pass y or None.

        """

        # data frame check
        X = _is_dataframe(X)

        # get input shape
        self._input_shape_ = X.shape

        # constant features
        self.constant_features_ = [feat for feat in X.columns if X[feat].nunique() == 1]

        # non-constant features
        self.selected_features_ = [feat for feat in X.columns if feat not in self.constant_features_]

        # if total constant features is equal to total features raise an error
        if len(self.constant_features_) == len(X.columns):
            raise ValueError("The resulting dataframe will have no columns after dropping all constant features")

        if self.drop_quasi_constant_features:
            # quasi-constant features
            self.quasi_constant_features_ = [feat for feat in X.columns if X[feat].value_counts(
                normalize=True, sort=True, ascending=False).iloc[0] > self.quasi_constant_threshold
                                             and feat not in self.constant_features_
                                             and feat in self.selected_features_]

            self.selected_features_ = [feat for feat in self.selected_features_
                                       if feat not in self.quasi_constant_features_]

        if len(self.constant_features_) + len(self.quasi_constant_features_) == len(X.columns):
            raise ValueError("The resulting dataframe will have no columns after dropping all constant "
                             "and quasi constant features")

        return self

    def transform(self, X):
        """
        Drops the constant and non-quasi constant features from a dataframe and returns the dataframe with
        selected features.

        Parameters
        ----------
        X: pandas dataframe
            The input dataframe from which features will be dropped

        Returns
        -------
        X_transformed: pandas dataframe of shape = [n_samples, n_features - (constant_features+quasi constant features)]
            The transformed dataframe with the remaining subset of variables.

        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self._input_shape_[1])

        # returned selected features
        X = X[self.selected_features_].copy()

        return X


