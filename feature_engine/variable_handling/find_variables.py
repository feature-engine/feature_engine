"""Functions to select certain types of variables."""

import warnings
from typing import List, Tuple, Union

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
    _is_categorical_and_is_not_datetime,
    _is_date_or_datetime,
)

# columns of these narwhals dtypes are candidates for being "categorical" - they
# still need to be run through the datetime-disambiguation helpers, because a
# String/Categorical column may actually hold dates. `Object` covers pandas
# columns holding a genuine mix of Python objects (e.g. strings and numbers),
# which narwhals cannot classify as String or any other single dtype.
_CATEGORICAL_CANDIDATE_SELECTOR = (
    nw.selectors.categorical()
    | nw.selectors.enum()
    | nw.selectors.string()
    | nw.selectors.by_dtype(nw.Object)
)


def _categorical_candidates_in_order(nw_df) -> List[Union[str, int]]:
    # nw.selectors' `|` combination returns matches grouped by which
    # sub-selector matched (e.g. all categorical() hits, then all string()
    # hits) rather than preserving the dataframe's original column order, for
    # the pandas backend specifically. Re-sort against the dataframe's own
    # column order so results are deterministic and backend-independent.
    matched = set(nw_df.select(_CATEGORICAL_CANDIDATE_SELECTOR).columns)
    return [column for column in nw_df.columns if column in matched]


def find_numerical_variables(
    X: IntoDataFrame,
    return_empty: bool = False,
) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the numerical variables in a dataframe.

    More details in the :ref:`User Guide <find_num_vars>`.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The dataset. Can be a pandas, polars, or any other dataframe supported by
        narwhals.

    return_empty : bool, default=False
        Whether to return an empty list when no numerical variables are found.
        If False, the function raises an error.

        .. versionadded:: 2.0
           `return_empty` currently defaults to False. The default will change to
           True in version 2.1. To keep the current behaviour and silence the
           warning, explicitly set `return_empty=False` instead of relying on the
           default.

    Returns
    -------
    variables: List
        The names of the numerical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_numerical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = find_numerical_variables(X)
    >>> var_
    ['var_num']
    """
    if nwd.is_pandas_dataframe(X):
        variables = list(X.select_dtypes(include="number").columns)
    else:
        nw_X = nw.from_native(X, eager_only=True)
        variables = list(nw_X.select(nw.selectors.numeric()).columns)

    if len(variables) == 0:
        if return_empty is False:
            raise TypeError(
                "No numerical variables found in this dataframe. Check variable "
                "dtypes or set return_empty to True to return an empty list instead."
            )
        else:
            warnings.warn(
                "No numerical variables found in this dataframe. "
                "Returning an empty list.",
                UserWarning,
            )
    return variables


def find_categorical_variables(
    X: IntoDataFrame,
    return_empty: bool = False,
) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the categorical variables in a dataframe.
    Note that variables cast as object that can be parsed to datetime will be
    excluded.

    More details in the :ref:`User Guide <find_cat_vars>`.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The dataset. Can be a pandas, polars, or any other dataframe supported by
        narwhals.

    return_empty : bool, default=False
        Whether to return an empty list when no categorical variables are found.
        If False, the function raises an error.

        .. versionadded:: 2.0
           `return_empty` currently defaults to False. The default will change to
           True in version 2.1. To keep the current behaviour and silence the
           warning, explicitly set `return_empty=False` instead of relying on the
           default.

    Returns
    -------
    variables: List
        The names of the categorical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_categorical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = find_categorical_variables(X)
    >>> var_
    ['var_cat']
    """
    nw_X = nw.from_native(X, eager_only=True)
    candidates = _categorical_candidates_in_order(nw_X)
    variables = [
        column
        for column in candidates
        if _is_categorical_and_is_not_datetime(nw_X.get_column(column))
    ]

    if len(variables) == 0:
        if return_empty is False:
            raise TypeError(
                "No categorical variables found in this dataframe. Check variable "
                "dtypes or set return_empty to True to return an "
                "empty list instead."
            )
        else:
            warnings.warn(
                "No categorical variables found in this dataframe. "
                "Returning an empty list.",
                UserWarning,
            )
    return variables


def find_datetime_variables(
    X: IntoDataFrame,
    return_empty: bool = False,
) -> List[Union[str, int]]:
    """
    Returns a list with the names of the variables that are or can be parsed as
    datetime.

    Note that this function will select variables cast as object if they can be cast as
    datetime as well.

    More details in the :ref:`User Guide <find_datetime_vars>`.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The dataset. Can be a pandas, polars, or any other dataframe supported by
        narwhals.

    return_empty : bool, default=False
        Whether to return an empty list when no datetime variables are found.
        If False, the function raises an error.

        .. versionadded:: 2.0
           `return_empty` currently defaults to False. The default will change to
           True in version 2.1. To keep the current behaviour and silence the
           warning, explicitly set `return_empty=False` instead of relying on the
           default.

    Returns
    -------
    variables: List
        The names of the datetime variables.

    Notes
    -----
    String columns are parsed with flexible, dateutil-backed date guessing, so
    formats like "01-Jan-2010" or "10/11/12" are recognised, in addition to
    ISO-8601 strings and native `Date`/`Datetime` columns, regardless of the
    dataframe library backing `X`.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_datetime_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_date = find_datetime_variables(X)
    >>> var_date
    ['var_date']
    """
    nw_X = nw.from_native(X, eager_only=True)
    non_numeric = [
        column for column in nw_X.columns if not nw_X.schema[column].is_numeric()
    ]
    variables = [
        column
        for column in non_numeric
        if _is_date_or_datetime(nw_X.schema[column])
        or _is_categorical_and_is_datetime(nw_X.get_column(column))
    ]

    if len(variables) == 0:
        if return_empty is False:
            raise TypeError(
                "No datetime variables found in this dataframe. To return an empty "
                "list instead of the error set return_empty to True."
            )
        else:
            warnings.warn(
                "No datetime variables found in this dataframe. "
                "Returning an empty list.",
                UserWarning,
            )
    return variables


def find_all_variables(
    X: IntoDataFrame,
    exclude_datetime: bool = False,
    return_empty: bool = False,
) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the variables in the dataframe.
    Optionally, it excludes variables that can be parsed as datetime or datetimetz.

    More details in the :ref:`User Guide <find_all_vars>`.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The dataset. Can be a pandas, polars, or any other dataframe supported by
        narwhals.

    exclude_datetime: bool, default=False
        Whether to exclude datetime variables.

    return_empty : bool, default=False
        Whether to return an empty list when no variables are found. If False, the
        function raises an error.

        .. versionadded:: 2.0
           `return_empty` currently defaults to False. The default will change to
           True in version 2.1. To keep the current behaviour and silence the
           warning, explicitly set `return_empty=False` instead of relying on the
           default.

    Returns
    -------
    variables: List
        The names of the variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_all_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> vars_all = find_all_variables(X)
    >>> vars_all
    ['var_num', 'var_cat', 'var_date']
    """
    nw_X = nw.from_native(X, eager_only=True)
    if exclude_datetime is True:
        variables = [
            var for var in nw_X.columns if not _is_date_or_datetime(nw_X.schema[var])
        ]
        variables = [
            var
            for var in variables
            if nw_X.schema[var].is_numeric()
            or not _is_categorical_and_is_datetime(nw_X.get_column(var))
        ]
    else:
        variables = list(nw_X.columns)

    if len(variables) == 0:
        if return_empty is False:
            raise TypeError(
                "No variables found in this dataframe. Set return_empty to "
                "True to return an empty list instead of the error."
            )
        else:
            warnings.warn(
                "No variables found in this dataframe. Returning an empty list.",
                UserWarning,
            )
    return variables


def find_categorical_and_numerical_variables(
    X: IntoDataFrame,
    variables: Union[None, int, str, List[Union[str, int]]] = None,
    return_empty: bool = False,
) -> Tuple[List[Union[str, int]], List[Union[str, int]]]:
    """
    Find numerical and categorical variables in a dataframe or from a list.

    The function returns two lists: the first with categorical variables and
    the second with numerical variables.

    More details in the :ref:`User Guide <find_cat_and_num_vars>`.

    Parameters
    ----------
    X : dataframe of shape = [n_samples, n_features]
        The dataset. Can be a pandas, polars, or any other dataframe supported by
        narwhals.

    variables : list, default=None
        If `None`, the function finds all categorical and numerical variables in X.
        Alternatively, it finds categorical and numerical variables in X, selecting
        from the given list.

    return_empty : bool, default=False
        Whether to return empty lists when no variables are found. If False, the
        function raises an error.

        .. versionadded:: 2.0
           `return_empty` currently defaults to False. The default will change to
           True in version 2.1. To keep the current behaviour and silence the
           warning, explicitly set `return_empty=False` instead of relying on the
           default.

    Returns
    -------
    variables: tuple
        Tuple containing a list with the categorical variables and a list with the
        numerical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import (
    >>>   find_categorical_and_numerical_variables
    >>>)
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_cat, var_num = find_categorical_and_numerical_variables(X)
    >>> var_cat, var_num
    (['var_cat'], ['var_num'])
    """
    nw_X = nw.from_native(X, eager_only=True)

    # If the user passes just 1 variable outside a list.
    if isinstance(variables, (str, int)):
        s = nw_X.get_column(variables)
        is_cat = isinstance(
            s.dtype, (nw.Categorical, nw.Enum)
        ) or _is_categorical_and_is_not_datetime(s)
        is_num = s.dtype.is_numeric()

        if is_cat:
            variables_cat = [variables]
            variables_num = []
        elif is_num:
            variables_num = [variables]
            variables_cat = []
        else:
            if return_empty is False:
                raise TypeError(
                    "The variable entered is neither numerical nor categorical. "
                    "Set return_empty to True to return empty lists instead of the "
                    "error."
                )
            else:
                warnings.warn(
                    "The variable entered is neither numerical nor "
                    "categorical. Returning empty lists.",
                    UserWarning,
                )
                variables_cat = []
                variables_num = []

    # If user leaves default None parameter.
    elif variables is None:
        candidates = _categorical_candidates_in_order(nw_X)
        variables_cat = [
            column
            for column in candidates
            if _is_categorical_and_is_not_datetime(nw_X.get_column(column))
        ]
        variables_num = list(nw_X.select(nw.selectors.numeric()).columns)

        if len(variables_num) == 0 and len(variables_cat) == 0:
            if return_empty is False:
                raise TypeError(
                    "There are no numerical or categorical variables in the dataframe. "
                    "Set return_empty to True to return empty lists instead of the "
                    "error"
                )
            else:
                warnings.warn(
                    "There are no numerical or categorical variables in the "
                    "dataframe. Returning empty lists.",
                    UserWarning,
                )
                variables_cat = []
                variables_num = []

    # If user passes variable list.
    else:
        if len(variables) == 0:
            if return_empty is False:
                raise ValueError(
                    "The list of variables provided is empty. If this was intentional "
                    "and you want to return empty lists set return_empty to True."
                )
            else:
                warnings.warn(
                    "The list of variables provided is empty. Returning "
                    "empty lists.",
                    UserWarning,
                )
                variables_cat = []
                variables_num = []

        else:
            sub_X = nw_X.select(variables)
            candidates = _categorical_candidates_in_order(sub_X)
            variables_cat = [
                column
                for column in candidates
                if _is_categorical_and_is_not_datetime(sub_X.get_column(column))
            ]
            variables_num = list(sub_X.select(nw.selectors.numeric()).columns)

    return variables_cat, variables_num
