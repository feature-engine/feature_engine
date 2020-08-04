from sklearn.base import BaseEstimator, TransformerMixin
import sklearn
from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import _define_variables, _find_all_variables, _find_numerical_variables
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers like the
    SimpleImputer() or OrdinalEncoder(), to allow the use of the
    transformer on a selected group of variables.

    Parameters
    ----------

    variables : list, default=None
        The list of variables to be imputed. If None, the wrapper will select
        all variables.

    transformer : sklearn transformer, default=None
        The desired Scikit-learn transformer.
    """

    def __init__(self, variables=None, transformer=None):
        self.variables = _define_variables(variables)
        self.transformer = transformer
        if isinstance(self.transformer, OneHotEncoder) and self.transformer.sparse:
            raise AttributeError('OneHotEncoder is available only with sparse=False attribute value.')

    def fit(self, X, y=None):
        """
        The `fit` method allows scikit-learn transformers to
        learn the required parameters from the training data set.

        Only numerical variables are transformed if transformer is StandardScaler, RobustScaler or MinMaxScaler.
        In other cases, all variables passed in variables parameter are transformed.
        If variables parameter is None, all variables existing in dataset are transformed.
        """

        # check input dataframe
        X = _is_dataframe(X)

        if isinstance(self.transformer, (sklearn.preprocessing.StandardScaler, sklearn.preprocessing.RobustScaler,
                                         sklearn.preprocessing.MinMaxScaler)):
            self.variables = _find_numerical_variables(X, self.variables)
        else:
            self.variables = _find_all_variables(X, self.variables)

        self.transformer.fit(X[self.variables])

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Apply the transformation to the dataframe.

        If transformer is OneHotEncoder, dummy features are concatenated to source dataset.
        In other cases features are transformed in-place.
        """

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input data contains same number of columns than
        # the dataframe used to fit the imputer.

        _check_input_matches_training_df(X, self.input_shape_[1])

        if isinstance(self.transformer, sklearn.preprocessing.OneHotEncoder):
            ohe_results_as_df = pd.DataFrame(
                data=self.transformer.transform(X[self.variables]),
                columns=self.transformer.get_feature_names(self.variables)
            )
            X = pd.concat([X, ohe_results_as_df], axis=1)
        else:
            X[self.variables] = self.transformer.transform(X[self.variables])

        return X

