""" The base transformer provides functionality that is shared by most transformer
classes. Provides the base functionality within the fit() and transform() methods
shared by most transformers, like checking that input is a df, the size, NA, etc.
"""

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import _find_or_check_datetime_variables


class DateTimeBaseTransformer(BaseEstimator, TransformerMixin):
    """shared set-up procedures across datetime transformers"""

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Checks that input is a dataframe, finds datetime variables,
        or alternatively checks that variables entered by the user are
        or can be converted to type datetime

        Parameters
        ----------
        X : Pandas DataFrame

        y : Pandas Series, np.array. Default = None
            Parameter is necessary for compatibility with sklearn Pipeline.

        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered as parameter
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for datetime variables
        self.variables_ = _find_or_check_datetime_variables(X, self.variables)

        # check if datetime variables contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA and Inf.

        Parameters
        ----------
        X : Pandas DataFrame

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

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)

        X = pd.concat(
            [
                pd.to_datetime(X[variable])
                if variable in self.variables_
                else X[variable]
                for variable in X.columns
            ],
            axis=1,
        )

        return X

    # for the check_estimator tests
    def _more_tags(self):
        return _return_tags()
