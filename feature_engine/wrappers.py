from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import _define_variables


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers like the
    SimpleImputer() or OrdinalEncoder(), to allow the use of the
    transformer on a selected group of variables.

    Parameters
    ----------

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will select
        all variables of type numeric.

    transformer : sklearn transformer, default=None
        The desired Scikit-learn transformer.
    """

    def __init__(self, variables=None, transformer=None):
        self.variables = _define_variables(variables)
        self.transformer = transformer

    def fit(self, X, y=None):
        """
        The `fit` method allows scikit-learn transformers to
        learn the required parameters from the training data set.
        """

        # check input dataframe
        X = _is_dataframe(X)

        self.transformer.fit(X[self.variables])

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """Apply the transformation to the dataframe."""

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input data contains same number of columns than
        # the dataframe used to fit the imputer.

        _check_input_matches_training_df(X, self.input_shape_[1])

        X[self.variables] = self.transformer.transform(X[self.variables])

        return X
