""" The base transformer provides functionality that is shared by most transformer
classes. This base transformer provides the base functionality within the fit() and
transform() methods shared by most transformers, like checking that input is a df,
the size, NA, etc.

"""
from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_contains_inf,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.variable_manipulation import _find_or_check_numerical_variables


class BaseNumericalTransformer(BaseEstimator, TransformerMixin):
    """shared set-up procedures across numerical transformers, i.e.,
    variable transformers, discretisers, math combination.
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Checks that input is a dataframe, finds numerical variables, or alternatively
        checks that variables entered by the user are of type numerical.

        Parameters
        ----------
        X : Pandas DataFrame

        y : Pandas Series, np.array. Default = None
            Parameter is necessary for compatibility with sklearn.pipeline.Pipeline.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
            If any of the user provided variables are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty
            If the variable(s) contain null values

        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered as parameter
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables_: List[Union[str, int]] = _find_or_check_numerical_variables(
            X, self.variables
        )

        # check if dataset contains na
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X : Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            If the variable(s) contain null values
            If the dataframe not of the same size as that used in fit()

        Returns
        -------
        X : Pandas DataFrame.
            The same dataframe entered by the user.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X

    # for the check_estimator tests
    def _more_tags(self):
        # to overcome the fact that sklearn does not allow nan in the input
        # to overcome the error thrown when transformer can't handle sparse data
        # to overcome tests that work only on numpy arrays and are thus not relevant
        # for feature-engine transformers
        return {
            '_xfail_checks': {
                # these are important checks that at the moment can't be run
                # because the input arrays contain 0s as values, and some transformers
                # return errors on those values, intentionally
                'check_estimators_dtypes':
                    'transformers raise errors when data contains zeroes',
                'check_estimators_fit_returns_self':
                    'transformers raise errors when data contains zeroes',
                'check_pipeline_consistency':
                    'transformers raise errors when data contains zeroes',
                'check_complex_data':
                    'I dont think we need this check',
                'check_dtype_object':
                    'Feature-engine uses dtypes to select variable types',
                'check_estimator_sparse_data':
                    'Feature-engine transformers do not work with sparse data',
                'check_transformer_data_not_an_array':
                    'Not sure what this check is at the moment',
                'check_transformer_preserve_dtypes':
                    'Feature-engine transformers can change the types',
                'check_methods_sample_order_invariance':
                    'Test does not work on dataframes',
                'check_fit_idempotent': 'Test does not work on dataframes',
                'check_fit1d': 'Feature-engine transformers only work with dataframes',
                'check_fit2d_predict1d':
                    'Feature-engine transformers only work with dataframes',
            }
        }
