from typing import Dict, List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_missing_values,
)
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

           Name  City  Age  Marks
        0   tom   NaN   20    0.9
        1   sam   NaN   22    0.7
        2  nick   NaN   23    0.6

    The order of the variables in the transformed dataset is also adjusted to match
    that observed in the train set.

    More details in the :ref:`User Guide <match_variables>`.

    Parameters
    ----------
    fill_value: integer, float or string. Default=np.nan
        The values for the variables that will be added to the transformed dataset.
        With polars dataframes, np.nan adds the variables as nulls.

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
    ...                             x2 = [5,6,4,7],
    ...                             x3 = [1,1,1,1]))
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
    ...                             x2 = [4,5,6], x3 = [1,1,1]))
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

    With polars:

    >>> import polars as pl
    >>> from feature_engine.preprocessing import MatchVariables
    >>> X_train = pl.DataFrame(dict(x1 = ["a","b","c"],
    ...                             x2 = [4,5,6], x3 = [1,1,1]))
    >>> X_test = pl.DataFrame(dict(x2 = [5,6,4,7],
    ...                            x1 = ["c","b","a","d"],
    ...                            x4 = [0,0,0,0]))
    >>> mv = MatchVariables(missing_values="ignore")
    >>> mv.fit(X_train)
    >>> mv.transform(X_test)
    The following variables are added to the DataFrame: ['x3']
    The following variables are dropped from the DataFrame: ['x4']
    shape: (4, 3)
    ┌─────┬─────┬──────┐
    │ x1  ┆ x2  ┆ x3   │
    │ --- ┆ --- ┆ ---  │
    │ str ┆ i64 ┆ f64  │
    ╞═════╪═════╪══════╡
    │ c   ┆ 5   ┆ null │
    │ b   ┆ 6   ┆ null │
    │ a   ┆ 4   ┆ null │
    │ d   ┆ 7   ┆ null │
    └─────┴─────┴──────┘
    """

    def __init__(
        self,
        fill_value: Union[str, int, float] = np.nan,
        missing_values: str = "raise",
        match_dtypes: bool = False,
        verbose: bool = True,
    ):
        _check_param_missing_values(missing_values)

        if not isinstance(match_dtypes, bool):
            raise ValueError(
                "match_dtypes takes only booleans True and False. "
                f"Got {match_dtypes} instead."
            )

        if not isinstance(verbose, bool):
            raise ValueError(
                f"verbose takes only booleans True and False. Got {verbose} instead."
            )

        # note: np.nan is an instance of float!!!
        if not isinstance(fill_value, (str, int, float)):
            raise ValueError(
                "fill_value takes integers, floats or strings. "
                f"Got {fill_value} instead."
            )

        self.fill_value = fill_value
        self.missing_values = missing_values
        self.match_dtypes = match_dtypes
        self.verbose = verbose

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """Learns and stores the names of the variables in the training dataset.

        Parameters
        ----------

        X: dataframe of shape = [n_samples, n_features]
            The input dataframe.

        y: None
            y is not needed for this transformer. You can pass y or None.
        """
        nw_X = check_X(X)

        if self.missing_values == "raise":
            _check_contains_na(X, nw_X.columns)

        self.feature_names_in_: List[Union[str, int]] = nw_X.columns
        self.n_features_in_ = nw_X.shape[1]

        if self.match_dtypes is True:
            # narwhals dtypes don't carry the categories of pandas categoricals.
            if nwd.is_pandas_dataframe(X) is True:
                self._dtype_dict: Dict = X.dtypes.to_dict()
            else:
                self._dtype_dict = dict(nw_X.schema)

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Drops variables that were not seen in the train set and adds variables that
        were in the train set but not in the data to transform. In other words, it
        returns a dataframe with matching columns.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features]
            The dataframe with variables that match those observed in the train set.
        """
        check_is_fitted(self)

        nw_X = check_X(X)
        columns = set(nw_X.columns)

        if self.missing_values == "raise":
            # Variables from the train set may be missing from X.
            _check_contains_na(
                X, [var for var in self.feature_names_in_ if var in columns]
            )

        train_columns = set(self.feature_names_in_)
        _columns_to_add = [var for var in self.feature_names_in_ if var not in columns]
        _columns_to_drop = [var for var in nw_X.columns if var not in train_columns]

        if self.verbose is True:
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

        if nwd.is_pandas_dataframe(X) is True:
            # pandas is faster than narwhals.
            X = X.reindex(columns=self.feature_names_in_, fill_value=self.fill_value)
            if self.match_dtypes is True:
                X = self._match_dtypes_pandas(X)
            return X

        if len(_columns_to_add) > 0:
            fill_value = self._fill_value_expression()
            nw_X = nw_X.with_columns(fill_value.alias(var) for var in _columns_to_add)
        nw_X = nw_X.select(self.feature_names_in_)

        if self.match_dtypes is True:
            nw_X = self._match_dtypes_narwhals(nw_X)

        return nw_X.to_native()

    def _fill_value_expression(self) -> nw.Expr:
        if isinstance(self.fill_value, float) and np.isnan(self.fill_value):
            # polars treats NaN as a value, not as missing data.
            return nw.lit(None, dtype=nw.Float64())
        if isinstance(self.fill_value, int) and not isinstance(self.fill_value, bool):
            # polars would store integers as Int32, pandas uses int64.
            return nw.lit(self.fill_value, dtype=nw.Int64())
        return nw.lit(self.fill_value)

    def _dtypes_to_update(self, current_dtypes: Dict) -> Dict:
        dtypes_to_update = {
            column: new_dtype
            for column, new_dtype in self._dtype_dict.items()
            if new_dtype != current_dtypes[column]
        }
        if self.verbose is True:
            for column, new_dtype in dtypes_to_update.items():
                print(
                    f"The {column} dtype is changing from ",
                    f"{current_dtypes[column]} to {new_dtype}",
                )
        return dtypes_to_update

    def _match_dtypes_pandas(self, X):
        dtypes_to_update = self._dtypes_to_update(X.dtypes.to_dict())

        for column, new_dtype in dtypes_to_update.items():
            # Handle pandas 4 future warning
            if new_dtype.name == "category":
                X[column] = X[column].where(X[column].isin(new_dtype.categories))
            elif new_dtype.kind in "iub" and X[column].hasnans is True:
                # numpy integers can't hold NaN and booleans turn it into True.
                dtypes_to_update[column] = self._nullable_dtype(new_dtype)

        return X.astype(dtypes_to_update)

    def _nullable_dtype(self, dtype) -> str:
        if dtype.kind == "b":
            return "boolean"
        prefix = "UInt" if dtype.kind == "u" else "Int"
        return f"{prefix}{dtype.itemsize * 8}"

    def _match_dtypes_narwhals(self, nw_X: nw.DataFrame) -> nw.DataFrame:
        current_dtypes = nw_X.schema
        dtypes_to_update = self._dtypes_to_update(current_dtypes)
        if len(dtypes_to_update) == 0:
            return nw_X

        expressions = []
        for column, new_dtype in dtypes_to_update.items():
            expression = nw.col(column)
            if isinstance(new_dtype, nw.Enum):
                # polars raises on values outside the categories, pandas sets NaN.
                # Strings, because is_in on an Enum rejects values it doesn't have.
                expression = expression.cast(nw.String())
                expression = nw.when(expression.is_in(new_dtype.categories)).then(
                    expression
                )
            elif current_dtypes[column] == nw.String:
                # polars does not parse strings when casting them to dates.
                if new_dtype == nw.Datetime:
                    expression = expression.str.to_datetime()
                elif new_dtype == nw.Date:
                    expression = expression.str.to_date()
            expressions.append(expression.cast(new_dtype))

        return nw_X.with_columns(expressions)

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
