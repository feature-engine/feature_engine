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


class DropConstantFeatures(TransformerMixin, BaseEstimator):
    """
    Drops constant and quasi-constant variables from a dataframe. Constant variables show the same value across all the
    observations in the dataset. Quasi-constant variables show the same value in almost all the observations in the
    dataset.

    By default, DropConstantFeatures drops only constant variables. This transformer works with both numerical and
    categorical variables. The user can indicate a list of variables to examine. Alternatively, the transformer will
    evaluate all the variables in the dataset.

    The transformer will first identify and store the constant and quasi-constant variables. Next, the transformer
    will drop these variables from a dataframe.

    Parameters
    ----------

    tol: float, default=1
        Threshold to detect constant/quasi-constant features. Variables showing the same value in a percentage of
        observations greater than tol will be considered constant / quasi-constant and dropped.

    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all variables in the dataset.
    """

    def __init__(self, tol=1, variables=None):

        if tol < 0 or tol > 1:
            raise ValueError("tol takes values between 0 and 1")

        self.tol = tol
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):

        """
        Find constant and quasi-constant features.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.

        y: None
            y is not needed for this transformer. You can pass y or None.


        Attributes
        ----------

        constant_features_: list
            The list of constant and quasi-constant features.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are present in the dataframe
        self.variables = _find_all_variables(X, self.variables)

        # find constant and quasi-constant
        self.constant_features_ = []

        for feature in self.variables:

            predominant = (X[feature].value_counts() / np.float(len(X))).sort_values(ascending=False).values[0]

            if predominant >= self.tol:
                self.constant_features_.append(feature)

        # if total constant features is equal to total features raise an error
        if len(self.constant_features_) == len(X.columns):
            raise ValueError("The resulting dataframe will have no columns after dropping all constant features.")

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Drops the constant and quasi-constant features from a dataframe.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input samples.

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
        X = X.drop(columns=self.constant_features_)

        return X
