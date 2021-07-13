from typing import Any, List
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _is_dataframe,
)
from feature_engine.variable_manipulation import _find_all_variables

class SimilarColumns(BaseEstimator, TransformerMixin):
    """Ensure that similar columns are in test and train dataset.


    Parameters
    ----
    None

    """

    def __init__(
        self,
        fill_value: Any = np.nan,
        missing_values: str = "raise",
        drop_extra_variables: bool = True,
        add_missing_variables: bool = True,
        verbose: bool = True
    ):

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'.")

        if not isinstance(drop_extra_variables, bool):
            raise ValueError("drop_extra_variables takes only booleans True and False")

        if not isinstance(add_missing_variables, bool):
            raise ValueError("add_missing_variables takes only booleans True and False")

        if not isinstance(verbose, bool):
            raise ValueError("verbose takes only booleans True and False")

        self.fill_value = fill_value
        self.missing_values = missing_values
        self.drop_extra_variables = drop_extra_variables
        self.add_missing_variables = add_missing_variables
        self.verbose = verbose

    def _check_input(self, X: pd.DataFrame):
        X = _is_dataframe(X)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        return X

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit columns schema

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe

        y: None
            y is not needed for this transformer. You can pass y or None.

        """
        self.n_features_in_ = X.shape[1]

        self.variables_ = X.columns
        X = self._check_input(X)

        
        return self

    def _raise_errors_if_needed(
        self,
        _columns_to_drop: List[str],
        _columns_to_add: List[str],
        X: pd.DataFrame
    ):

        if not self.drop_extra_variables and len(_columns_to_drop) > 0:
            raise ValueError(
                "drop_extra_variables is set to False"
                " and transform df contain more columns "
                "than the df used in fit "
                f"(found {set(X.columns)}, "
                f"expected ({set(self.variables_)}))."
            )

        if not self.add_missing_variables and len(_columns_to_add) > 0:
            raise ValueError(
                "add_missing_variables is set to False"
                " and transform df contain less columns "
                "than the df used in fit "
                f"(found {set(X.columns)}, "
                f"expected ({set(self.variables_)}))."
            )

    def transform(self, X: pd.DataFrame, **transform_params) -> pd.DataFrame:
        """Drops the variable that are not in the fitted dataframe and returns
        a new dataframe with the remaining subset of variables.

        If a column is in train but not in test, then the column will be created in
        test dataset with missing value everywhere.

        If a column is in test but not in train, it will be dropped.

        Parameters
        ----------
        X: pandas dataframe
            The input dataframe from which features will be dropped

        Returns
        -------
        X_transformed: pandas dataframe of shape =
             [n_samples, n_features - len(features_to_drop)]

                The transformed dataframe with the same columns
                (in same order) than the fitted dataframe

        """
        check_is_fitted(self)

        X = self._check_input(X)

        _columns_to_drop = list(set(X.columns) - set(self.variables_))
        _columns_to_add = list(set(self.variables_) - set(X.columns))

        self._raise_errors_if_needed(_columns_to_drop, _columns_to_add, X)

        if self.add_missing_variables:
            if self.verbose:
                logging.warning(
                    f"Columns : {_columns_to_add} are in fit but not"
                    " in transform, they will be added to the transform"
                    f"  dataframe with value {self.fill_value}"
                )

            X = X.reindex(
                    columns=list(X.columns) + _columns_to_add,
                    fill_value=self.fill_value
                )

        if self.drop_extra_variables:
            if self.verbose:
                logging.warning(
                    f"Columns ({_columns_to_drop}) are in transform df"
                    " but not in fit, they will be dropped"
                )

            X = X.drop(_columns_to_drop, axis=1)

        # reorder columns
        X = X.loc[:, self.variables_]
        return X
