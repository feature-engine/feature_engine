"""Functions to check that the variables in a list are of a certain type."""

from typing import List, Union

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
)

Variables = Union[int, str, List[Union[str, int]]]


def check_numerical_variables(
    X: IntoDataFrame, variables: Variables
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are of type numerical.

    More details in the :ref:`User Guide <check_num_vars>`.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The dataset. Can be a pandas, polars, or any other dataframe supported by
        narwhals.

    variables : List
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the numerical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_numerical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="min")
    >>> })
    >>> var_ = check_numerical_variables(X, variables=["var_num"])
    >>> var_
    ['var_num']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    if nwd.is_pandas_dataframe(X) is True:
        not_numerical = len(X[variables].select_dtypes(exclude="number").columns) > 0
    else:
        sub_X = nw.from_native(X, eager_only=True).select(variables)
        not_numerical = len(sub_X.select(nw.selectors.numeric()).columns) != len(
            sub_X.columns
        )

    if not_numerical is True:
        raise TypeError(
            "Some of the variables are not numerical. Please cast them as "
            "numerical before using this transformer."
        )

    return variables


def check_categorical_variables(
    X: IntoDataFrame, variables: Variables
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are of type object or categorical.

    More details in the :ref:`User Guide <check_cat_vars>`.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The dataset. Can be a pandas, polars, or any other dataframe supported by
        narwhals.

    variables : list
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the categorical variables.

    Notes
    -----
    For polars (and other non-pandas dataframes), plain string columns are
    accepted as categorical. Polars has no separate "object" dtype the way
    pandas does, so its `String` dtype is the only way to represent free-form
    text and is treated as categorical here.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_categorical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="min")
    >>> })
    >>> var_ = check_categorical_variables(X, "var_cat")
    >>> var_
    ['var_cat']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    if nwd.is_pandas_dataframe(X) is True:
        not_categorical = (
            len(
                X[variables]
                .select_dtypes(exclude=["O", "category", "string"])
                .columns
            )
            > 0
        )
    else:
        sub_X = nw.from_native(X, eager_only=True).select(variables)
        not_categorical = len(
            sub_X.select(
                nw.selectors.categorical() | nw.selectors.enum() | nw.selectors.string()
            ).columns
        ) != len(sub_X.columns)

    if not_categorical is True:
        raise TypeError(
            "Some of the variables are not categorical. Please cast them as "
            "object or categorical before using this transformer."
        )

    return variables


def check_datetime_variables(
    X: IntoDataFrame,
    variables: Variables,
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are or can be parsed as datetime and or
    datetimetz.

    More details in the :ref:`User Guide <check_datetime_vars>`.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The dataset. Can be a pandas, polars, or any other dataframe supported by
        narwhals.

    variables : list
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the datetime variables.

    Notes
    -----
    String columns are parsed with flexible, dateutil-backed date guessing, in
    addition to ISO-8601 strings and native `Date`/`Datetime` columns,
    regardless of the dataframe library backing `X`.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_datetime_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="min")
    >>> })
    >>> var_date = check_datetime_variables(X, "var_date")
    >>> var_date
    ['var_date']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    if nwd.is_pandas_dataframe(X) is True:
        sub_X = X[variables]
        candidates = sub_X.select_dtypes(exclude=["datetime", "datetimetz"]).columns
        numeric_cols = set(sub_X.select_dtypes(include="number").columns)
        nw_X = nw.from_native(sub_X, eager_only=True)
        non_datetime = any(
            column in numeric_cols
            or not _is_categorical_and_is_datetime(nw_X.get_column(column))
            for column in candidates
        )
    else:
        sub_X = nw.from_native(X, eager_only=True).select(variables)
        candidates = sub_X.select(~nw.selectors.by_dtype(nw.Date, nw.Datetime)).columns
        numeric_cols = set(sub_X.select(nw.selectors.numeric()).columns)
        non_datetime = any(
            column in numeric_cols
            or not _is_categorical_and_is_datetime(sub_X.get_column(column))
            for column in candidates
        )

    if non_datetime is True:
        raise TypeError(
            "Some of the variables are not or cannot be parsed as datetime."
        )

    return variables


def check_all_variables(
    X: IntoDataFrame,
    variables: Variables,
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are in the dataframe.

    More details in the :ref:`User Guide <check_all_vars>`.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The dataset. Can be a pandas, polars, or any other dataframe supported by
        narwhals.

    variables : list
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_all_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="min")
    >>> })
    >>> vars_all = check_all_variables(X, ['var_num', 'var_cat', 'var_date'])
    >>> vars_all
    ['var_num', 'var_cat', 'var_date']
    """
    if nwd.is_pandas_dataframe(X) is True:
        columns = set(X.columns)
    else:
        columns = set(nw.from_native(X, eager_only=True).columns)

    if isinstance(variables, (str, int)):
        if variables not in columns:
            raise KeyError(f"The variable {variables} is not in the dataframe.")
        variables_ = [variables]

    else:
        if set(variables).issubset(columns) is False:
            raise KeyError("Some of the variables are not in the dataframe.")

        variables_ = variables

    return variables_
