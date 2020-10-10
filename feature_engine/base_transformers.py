# Transformation methods are shared by most transformer groups.
# Each transformer can inherit the transform method from these base classes.

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (_check_contains_na,
                                             _check_input_matches_training_df,
                                             _is_dataframe)
from feature_engine.variable_manipulation import _find_numerical_variables


class BaseNumericalTransformer(BaseEstimator, TransformerMixin):
    # shared set-up procedures across numerical transformers, i.e.,
    # variable transformers, discretisers, math combination

    def fit(self, X: pd.DataFrame, y: Optional[str] = None) -> pd.DataFrame:
        """
        Fits the transformation to the DataFrame.

        Args:
            X: Pandas DataFrame to fit the transformation
            y: This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.
            Defaults to None.

        Returns:
            DataFrame with fitted transformation

        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies transformation to the DataFrame.

        Args:
            X: Pandas DataFrame to apply the transformation

        Returns:
            Transformed DataFrame
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X
