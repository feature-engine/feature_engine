""" The base transformer provides functionality that is shared by most transformer
classes. Provides the base functionality within the fit() and transform() methods
shared by most transformers, like checking that input is a df, the size, NA, etc.
"""
from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.validation import _return_tags
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
            Parameter is necessary for compatibility with sklearn Pipeline.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame or a numpy array
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

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA and Inf.

        Parameters
        ----------
        X : Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X : Pandas DataFrame.
            The same dataframe entered by the user.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        return X

    # for the check_estimator tests
    def _more_tags(self):
        return _return_tags()
