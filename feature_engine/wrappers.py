import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import _define_variables, _find_all_variables, _find_numerical_variables


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers like the
    SimpleImputer() or OrdinalEncoder(), to allow the use of the
    transformer on a selected group of variables.

    Parameters
    ----------

    variables : list, default=None
        The list of variables to be imputed.

        If None, the wrapper will select all variables of type numeric for all transformers
        except the SimpleImputer, OrdinalEncoder and OneHotEncoder, in which case it will
        select all variables in the dataset.

    transformer : sklearn transformer, default=None
        The desired Scikit-learn transformer.
    """

    def __init__(self, variables=None, transformer=None):
        self.variables = _define_variables(variables)
        self.transformer = transformer
        if isinstance(self.transformer, OneHotEncoder) and self.transformer.sparse:
            raise AttributeError('The SklearnTransformerWrapper can only wrap the OneHotEncoder if you '
                                 'set its sparse attribute to False')

    def fit(self, X, y=None):
        """
        The `fit` method allows Scikit-learn transformers to learn the required parameters
        from the training data set.

        If transformer is OneHotEncoder, OrdinalEncoder or SimpleImputer, all variables indicated
        in the variables parameter will be transformed. When the variables parameter is None, the
        SklearnWrapper will automatically select and transform all features in the dataset,
        numerical or otherwise.

        For all other Scikit-learn transformers only numerical variables will be transformed.
        The SklearnWrapper will check that the variables indicated in the variables parameter
        are numerical, or alternatively, if variables is None, it will automatically select
        the numerical variables in the data set.
        """

        # check input dataframe
        X = _is_dataframe(X)

        if isinstance(self.transformer, (OneHotEncoder, OrdinalEncoder, SimpleImputer)):
            self.variables = _find_all_variables(X, self.variables)
        else:
            self.variables = _find_numerical_variables(X, self.variables)

        self.transformer.fit(X[self.variables])

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Apply the transformation to the dataframe. Only the selected features will be modified. 

        If transformer is OneHotEncoder, dummy features are concatenated to the source dataset.
        Note that the original categorical variables will not be removed from the dataset
        after encoding. If this is the desired effect, please use Feature-engine's 
        OneHotCategoricalEncoder instead.
        """

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input data contains same number of columns than
        # the dataframe used to fit the imputer.

        _check_input_matches_training_df(X, self.input_shape_[1])

        if isinstance(self.transformer, OneHotEncoder):
            ohe_results_as_df = pd.DataFrame(
                data=self.transformer.transform(X[self.variables]),
                columns=self.transformer.get_feature_names(self.variables)
            )
            X = pd.concat([X, ohe_results_as_df], axis=1)
        else:
            X[self.variables] = self.transformer.transform(X[self.variables])

        return X

