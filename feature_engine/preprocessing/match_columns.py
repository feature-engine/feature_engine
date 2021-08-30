from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import _check_contains_na, _is_dataframe
from feature_engine.validation import _return_tags


class MatchColumnsToTrainSet(BaseEstimator, TransformerMixin):
    """
    MatchColumnsToTrainSet() ensures that the same variables observed in the train set
    are present in the test set. If the dataset to transform contains variables that
    were not present in the train set, they are dropped. If the dataset to transform
    lacks variables that were present in the train set, these variables are added to
    the dataframe with a value determined by the user (np.nan by default).

    .. code-block:: python

        train = pd.DataFrame({
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
        })

        test = pd.DataFrame({
            "Name": ["tom", "sam", "nick"],
            "Age": [20, 22, 23],
            "Marks": [0.9, 0.7, 0.6],
            "Hobbies": ["tennis", "rugby", "football"]
        })

        match_columns = MatchColumnsToTrainSet()

        match_columns.fit(train)

        df_transformed = match_columns.transform(test)

        df_transformed

            Name    City  Age  Marks
        0    tom  np.nan   20    0.9
        1    sam  np.nan   22    0.7
        2   nick  np.nan   23    0.6


    The order of the variables in the transformed dataset is also adjusted to match
    that observed in the train set.

    Parameters
    ----------
    fill_value: integer, float or string. Default=np.nan
        The values for the variables that will be added to the transformed dataset.

    missing_values: string, default='ignore'
        Indicates if missing values should be ignored or raised. If 'ignore', the
        transformer will ignore missing data when transforming the data. If 'raise' the
        transformer will return an error if the training or the datasets to transform
        contain missing values.

    verbose: bool, default=True
        If True, the transformer will print out the names of the variables that are
        added and / or removed from the dataset.

    Attributes
    ----------
    variables_:
        The variables present in the train set, in the order observed during fit.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Identify the variable names in the train set.
    transform:
        Add or delete variables to match those observed in the train set.
    fit_transform:
        Fit to the data. Then transform it.
    """

    def __init__(
        self,
        fill_value: Union[str, int, float] = np.nan,
        missing_values: str = "raise",
        verbose: bool = True,
    ):

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'."
                f"Got '{missing_values} instead."
            )

        if not isinstance(verbose, bool):
            raise ValueError(
                "verbose takes only booleans True and False." f"Got '{verbose} instead."
            )

        # note: np.nan is an instance of float!!!
        if not isinstance(fill_value, (str, int, float)):
            raise ValueError(
                "fill_value takes integers, floats or strings."
                f"Got '{fill_value} instead."
            )

        self.fill_value = fill_value
        self.missing_values = missing_values
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Learns and stores the names of the variables in the training dataset.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe

        y: None
            y is not needed for this transformer. You can pass y or None.

        Raises
        ------
        TypeError
           - If the input is not a Pandas DataFrame

        Returns
        -------
        self
        """
        X = _is_dataframe(X)

        if self.missing_values == "raise":
            # check if dataset contains na or inf
            _check_contains_na(X, X.columns)

        self.variables_ = list(X.columns)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drops variables that were not seen in the train set and adds variables that
         were in the train set but not in the data to transform.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_transformed: Pandas dataframe, shape = [n_samples, n_features]
             The dataframe with variables that match those observed in the train set.
        """
        check_is_fitted(self)

        X = _is_dataframe(X)

        if self.missing_values == "raise":
            # check if dataset contains na or inf
            _check_contains_na(X, self.variables_)

        _columns_to_drop = list(set(X.columns) - set(self.variables_))
        _columns_to_add = list(set(self.variables_) - set(X.columns))

        if self.verbose:
            if len(_columns_to_add) > 0:
                print(
                    "The following variables are added to the DataFrame: "
                    f"{_columns_to_add}"
                )
            if len(_columns_to_drop) > 0:
                print(
                    "The following variables are dropped from the DataFrame: "
                    f"{_columns_to_drop}"
                )

        X = X.drop(_columns_to_drop, axis=1)

        X = X.reindex(columns=self.variables_, fill_value=self.fill_value)

        return X

    # for the check_estimator tests
    def _more_tags(self):
        tags_dict = _return_tags()

        msg = "input shape of dataframes in fit and transform can differ"
        tags_dict["_xfail_checks"]["check_transformer_general"] = msg

        msg = (
            "transformer takes categorical varriables, and inf cannot be determined"
            "on these variables. Thus, check is not implemented"
        )
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = msg

        return tags_dict
