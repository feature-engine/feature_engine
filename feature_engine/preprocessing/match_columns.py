from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.validation import _return_tags

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _is_dataframe,
)
from feature_engine.variable_manipulation import _find_all_variables


class MatchColumnsToTrainSet(BaseEstimator, TransformerMixin):
    """Ensure that similar columns are in test and train dataset.

    Parameters
    ----
    None

    """

    def __init__(
        self,
        fill_value: Union[np.nan, int] = np.nan,
        missing_values: str = "raise",
        verbose: bool = True
    ):

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'.")

        if not isinstance(verbose, bool):
            raise ValueError("verbose takes only booleans True and False")
        
        if not (isinstance(fill_value, int) or np.isnan(fill_value)):
            raise ValueError("fill_value takes only int or np.nan")

        self.fill_value = fill_value
        self.missing_values = missing_values
        self.verbose = verbose

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
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
        X = _is_dataframe(X)

        self.n_features_in_ = X.shape[1]

        self.variables_ = _find_all_variables(X, list(X.columns))

        X = self._check_input(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

        X = X.reindex(
                columns=list(X.columns) + _columns_to_add,
                fill_value=self.fill_value
            )

        X = X.drop(_columns_to_drop, axis=1)

        # reorder columns
        X = X.loc[:, self.variables_]
        return X

    # for the check_estimator tests
    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["_xfail_checks"]["check_transformer_general"] = (
            "Transformer accept transform input shape to "
            "be different than fit input shape"
            )
        return tags_dict
