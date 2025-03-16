from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine.dataframe_checks import _check_contains_na, check_X
from feature_engine.tags import _return_tags


class MatchVariables(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    MatchVariables() ensures that the same variables observed in the train set
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

        match_columns = MatchVariables()

        match_columns.fit(train)

        df_transformed = match_columns.transform(test)

    Note that in the returned dataframe, the variable "Hobbies" was removed and the
    variable "City" was added with np.nan:

    .. code-block:: python

        df_transformed

            Name    City  Age  Marks
        0    tom  np.nan   20    0.9
        1    sam  np.nan   22    0.7
        2   nick  np.nan   23    0.6


    The order of the variables in the transformed dataset is also adjusted to match
    that observed in the train set.

    More details in the :ref:`User Guide <match_variables>`.

    Parameters
    ----------
    fill_value: integer, float or string. Default=np.nan
        The values for the variables that will be added to the transformed dataset.

    missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. If 'raise' the
        transformer will return an error if the datasets to `fit` or `transform`
        contain missing values. If 'ignore', missing data will be ignored when learning
        parameters or performing the transformation.

    match_dtypes: bool, default=False
        Indicates whether the dtypes observed in the train set should be applied to
        variables in the test set.

    verbose: bool, default=True
        If True, the transformer will print out the names of the variables that are
        added and / or removed from the dataset.

    Attributes
    ----------
    feature_names_in_:
        The variables present in the train set, in the order observed during fit.

    n_features_in_:
        The number of features in the train set used in fit.

    dtype_dict_:
        If `match_dtypes` is set to `True`, then this attribute will exist, and it will
        contain a dictionary of variables and their corresponding dtypes.

    Methods
    -------
    fit:
        Identify the variable names in the train set.

    fit_transform:
        Fit to the data. Then transform it.

    get_feature_names_out:
        Get output feature names for transformation.

    get_params:
        Get parameters for this estimator.

    set_params:
        Set the parameters of this estimator.

    transform:
        Add or delete variables to match those observed in the train set.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.preprocessing import MatchVariables
    >>> X_train = pd.DataFrame(dict(x1 = ["a","b","c"], x2 = [4,5,6]))
    >>> X_test = pd.DataFrame(dict(x1 = ["c","b","a","d"],
    >>>                             x2 = [5,6,4,7],
    >>>                             x3 = [1,1,1,1]))
    >>> mv = MatchVariables(missing_values="ignore")
    >>> mv.fit(X_train)
    >>> mv.transform(X_train)
    x1  x2
    0  a   4
    1  b   5
    2  c   6
    >>> mv.transform(X_test)
    The following variables are dropped from the DataFrame: ['x3']
      x1  x2
    0  c   5
    1  b   6
    2  a   4
    3  d   7

    >>> import pandas as pd
    >>> from feature_engine.preprocessing import MatchVariables
    >>> X_train = pd.DataFrame(dict(x1 = ["a","b","c"],
    >>>                             x2 = [4,5,6], x3 = [1,1,1]))
    >>> X_test = pd.DataFrame(dict(x1 = ["c","b","a","d"], x2 = [5,6,4,7]))
    >>> mv = MatchVariables(missing_values="ignore")
    >>> mv.fit(X_train)
    >>> mv.transform(X_train)
      x1  x2  x3
    0  a   4   1
    1  b   5   1
    2  c   6   1
    >>> mv.transform(X_test)
    The following variables are added to the DataFrame: ['x3']
      x1  x2  x3
    0  c   5 NaN
    1  b   6 NaN
    2  a   4 NaN
    3  d   7 NaN
    """

    def __init__(
        self,
        fill_value: Union[str, int, float] = np.nan,
        missing_values: str = "raise",
        match_dtypes: bool = False,
        verbose: bool = True,
    ):
        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'."
                f"Got '{missing_values} instead."
            )

        if not isinstance(match_dtypes, bool):
            raise ValueError(
                "match_dtypes takes only booleans True and False. "
                f"Got '{match_dtypes} instead."
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
        self.match_dtypes = match_dtypes
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Learns and stores the names of the variables in the training dataset.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.

        y: None
            y is not needed for this transformer. You can pass y or None.
        """
        X = check_X(X)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, X.columns)

        # save input features
        self.feature_names_in_: List[Union[str, int]] = X.columns.tolist()

        self.n_features_in_ = X.shape[1]

        if self.match_dtypes:
            self.dtype_dict_: Dict = X.dtypes.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops variables that were not seen in the train set and adds variables that
        were in the train set but not in the data to transform. In other words, it
        returns a dataframe with matching columns.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features]
             The dataframe with variables that match those observed in the train set.
        """
        check_is_fitted(self)

        X = check_X(X)

        if self.missing_values == "raise":
            # Some variables from the train set may not be present in the test set
            # and vice versa. We'll check for nan only in the variables seen during
            # training.
            vars = [var for var in self.feature_names_in_ if var in X.columns]
            _check_contains_na(X, vars)

        _columns_to_drop = list(set(X.columns) - set(self.feature_names_in_))
        _columns_to_add = list(set(self.feature_names_in_) - set(X.columns))

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

        X = X.reindex(columns=self.feature_names_in_, fill_value=self.fill_value)

        if self.match_dtypes:
            _current_dtypes = X.dtypes.to_dict()
            _columns_to_update = {
                column: new_dtype
                for column, new_dtype in self.dtype_dict_.items()
                if new_dtype != _current_dtypes[column]
            }

            if self.verbose:
                for column, new_dtype in _columns_to_update.items():
                    print(
                        f"The {column} dtype is changing from ",
                        f"{_current_dtypes[column]} to {new_dtype}",
                    )

            X = X.astype(_columns_to_update)

        return X

    # for the check_estimator tests
    def _more_tags(self):
        tags_dict = _return_tags()

        msg = "input shape of dataframes in fit and transform can differ"
        tags_dict["_xfail_checks"]["check_transformer_general"] = msg

        msg = (
            "transformer takes categorical variables, and inf cannot be determined"
            "on these variables. Thus, check is not implemented"
        )
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
