import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import _define_variables, _find_all_variables


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


class DropConstantAndQuasiConstantFeatures(TransformerMixin, BaseEstimator):
    """

    Drops constant and quasi constant variables from a dataframe.
    Constant variables are those variables which show only one value for all the observations.
    By default, all constant variables are dropped.
    Quasi-constant variables are those variables which are almost constant. In other words, those features which have
    the same value for a large subset of total observations.
    This transformer works for both numerical and categorical variables.

    Parameters
    ----------

    tol: float, default=1
        Threshold to detect constant/quasi-constant features. Variables showing tol percentage of values will be dropped

    variables: list, default=None
        The list of variables to inspect. If None, the transformer will select all variables in the dataframe

    Attributes
    ----------

    selected_features_: list
        List of selected features which are non-constant and non-quasi constant

    """

    def __init__(self, tol=1, variables=None):

        if tol < 0 or tol > 1:
            raise ValueError("tol takes values between 0 and 1")

        self.tol = tol
        self.variables = _define_variables(variables)

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

        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are present in the dataframe
        self.variables = _find_all_variables(X, self.variables)

        if self.tol == 1:

            # constant features
            constant_features = [feat for feat in self.variables if X[feat].nunique() == 1]

            self.selected_features_ = [feat for feat in X.columns if feat not in constant_features]

            # if total constant features is equal to total features raise an error
            if len(constant_features) == len(X.columns):
                raise ValueError("The resulting dataframe will have no columns after dropping all constant features")

        else:
            constant_plus_quasi_constant_features = []
            self.selected_features_ = []

            for feat in self.variables:

                predominant = (X[feat].value_counts() / np.float(len(X))).sort_values(ascending=False).values[0]

                if predominant > self.tol:
                    constant_plus_quasi_constant_features.append(feat)

            # if total constant features is equal to total features raise an error
            if len(constant_plus_quasi_constant_features) == len(X.columns):
                raise ValueError("The resulting dataframe will have no columns after dropping "
                                 "constant and quasi constant features")

            self.selected_features_ = [feat for feat in X.columns if feat not in constant_plus_quasi_constant_features]

        self.input_shape_ = X.shape

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
        _check_input_matches_training_df(X, self.input_shape_[1])

        # returned selected features
        X = X[self.selected_features_].copy()

        return X
