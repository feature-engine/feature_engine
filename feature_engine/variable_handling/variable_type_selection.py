"""Functions to select certain types of variables."""

from typing import List, Tuple, Union

import pandas as pd
from pandas.api.types import is_categorical_dtype as is_categorical
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_numeric_dtype as is_numeric
from pandas.api.types import is_object_dtype as is_object

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
    _is_categorical_and_is_not_datetime,
)

Variables = Union[None, int, str, List[Union[str, int]]]


def find_or_check_numerical_variables(
    X: pd.DataFrame, variables: Variables = None
) -> List[Union[str, int]]:
    """
    Returns the names of all the numerical variables in a dataframe. Alternatively, it
    checks that the variables entered by the user are numerical.

    More details in the :ref:`User Guide <find_num_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list, default=None
        If `None`, the function will return the names of all numerical variables in X.
        Alternatively, it checks that the variables in the list are of type numerical.

    Returns
    -------
    variables: List
        The names of the numerical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_or_check_numerical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_num = find_or_check_numerical_variables(X)
    >>> var_num

    ['var_num']
    """

    if variables is None:
        # find numerical variables in dataset
        variables = list(X.select_dtypes(include="number").columns)
        if len(variables) == 0:
            raise ValueError(
                "No numerical variables found in this dataframe. Please check "
                "variable format with pandas dtypes."
            )

    elif isinstance(variables, (str, int)):
        if is_numeric(X[variables]):
            variables = [variables]
        else:
            raise TypeError("The variable entered is not numeric.")

    else:
        if len(variables) == 0:
            raise ValueError("The list of variables is empty.")

        # check that user entered variables are of type numerical
        else:
            if len(X[variables].select_dtypes(exclude="number").columns) > 0:
                raise TypeError(
                    "Some of the variables are not numerical. Please cast them as "
                    "numerical before using this transformer."
                )

    return variables


def find_or_check_categorical_variables(
    X: pd.DataFrame, variables: Variables = None
) -> List[Union[str, int]]:
    """
    Returns the names of all the variables of type object or categorical in a dataframe.
    Alternatively, it checks that the variables entered by the user are of type object
    or categorical.

    Note that when `variables` is `None`, the transformer will not select variables of
    type object that can be parsed as datetime. But if the user passes a list with
    datetime variables cast as object to the `variables` parameter, they will be
    allowed.

    More details in the :ref:`User Guide <find_cat_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list, default=None
        If `None`, the function returns the names of all object or categorical variables
        in X. Alternatively, it checks that the variables in the list are of type
        object or categorical.

    Returns
    -------
    variables: List
        The names of the categorical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_or_check_categorical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_cat = find_or_check_categorical_variables(X)
    >>> var_cat
    ['var_cat']
    """

    if variables is None:
        # find categorical variables in dataset
        variables = [
            column
            for column in X.select_dtypes(include=["O", "category"]).columns
            if _is_categorical_and_is_not_datetime(X[column])
        ]
        if len(variables) == 0:
            raise ValueError(
                "No categorical variables found in this dataframe. Please check "
                "variable format with pandas dtypes."
            )

    elif isinstance(variables, (str, int)):
        if is_categorical(X[variables]) or is_object(X[variables]):
            variables = [variables]
        else:
            raise TypeError("The variable entered is not categorical.")

    else:
        if len(variables) == 0:
            raise ValueError("The list of variables is empty.")

        # check that user entered variables are of type categorical
        else:
            if len(X[variables].select_dtypes(exclude=["O", "category"]).columns) > 0:
                raise TypeError(
                    "Some of the variables are not categorical. Please cast them as "
                    "categorical or object before using this transformer."
                )

    return variables


def find_or_check_datetime_variables(
    X: pd.DataFrame, variables: Variables = None
) -> List[Union[str, int]]:
    """
    Returns the names of all the variables that are or can be parsed as datetime.
    Alternatively, it checks that the variables entered by the user can be parsed as
    datetime.

    Note that this function will select variables cast as object if they can be cast as
    datetime as well.

    More details in the :ref:`User Guide <find_datetime_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list, default=None
        If `None`, the function returns the names of all variables in X that can be
        parsed as datetime. These include those cast as datetime, and also object and
        categorical if they can be transformed to datetime variables. Alternatively, it
        checks that the variables in the list are or can be parsed to datetime.

    Returns
    -------
    variables: List
        The names of the datetime variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_or_check_datetime_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_date = find_or_check_datetime_variables(X)
    >>> var_date
    ['var_date']
    """

    if variables is None:
        variables = [
            column
            for column in X.select_dtypes(exclude="number").columns
            if is_datetime(X[column]) or _is_categorical_and_is_datetime(X[column])
        ]

        if len(variables) == 0:
            raise ValueError("No datetime variables found in this dataframe.")

    elif isinstance(variables, (str, int)):

        if is_datetime(X[variables]) or (
            not is_numeric(X[variables])
            and _is_categorical_and_is_datetime(X[variables])
        ):
            variables = [variables]
        else:
            raise TypeError("The variable entered is not datetime.")

    else:
        if len(variables) == 0:
            raise ValueError("The indicated list of variables is empty.")

        # check that the variables entered by the user are datetime
        else:
            vars_non_dt = [
                column
                for column in X[variables].select_dtypes(exclude="datetime")
                if is_numeric(X[column])
                or not _is_categorical_and_is_datetime(X[column])
            ]

            if len(vars_non_dt) > 0:
                raise TypeError("Some of the variables are not datetime.")

    return variables


def find_all_variables(
    X: pd.DataFrame,
    variables: Variables = None,
    exclude_datetime: bool = False,
) -> List[Union[str, int]]:
    """
    Returns the names of all the variables in the dataframe in a list. Alternatively,
    it will check that the variables indicated by the user are in the dataframe.

    More details in the :ref:`User Guide <find_all_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list, default=None
        If `None`, the function will return the names of all the variables in X.
        Alternatively, it checks that the variables in the list are present in the
        dataframe.

    exclude_datetime: bool, default=False
        Whether to exclude datetime variables.

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
    # find all variables in dataset
    if variables is None:
        if exclude_datetime is True:
            variables_ = X.select_dtypes(exclude="datetime").columns.to_list()
        else:
            variables_ = X.columns.to_list()

    elif isinstance(variables, (str, int)):
        if variables not in X.columns.to_list():
            raise ValueError(f"The variable {variables} is not in the dataframe.")
        variables_ = [variables]

    else:
        if len(variables) == 0:
            raise ValueError("The list of variables is empty.")

        if any(f for f in variables if f not in X.columns):
            raise KeyError("Some of the variables are not in the dataframe.")

        variables_ = variables

    return variables_


def _filter_out_variables_not_in_dataframe(X, variables):
    """Filter out variables that are not present in the dataframe.

    Function removes variables that the user defines in the argument `variables`
    but that are not present in the input dataframe.

    Useful when using several feature selection procedures in a row. The dataframe
    input to the first selection algorithm likely contains more variables than the
    input dataframe to subsequent selection algorithms, and it is not possible a
    priori, to say which variable will be dropped.

    Parameters
    ----------
    X:  pandas DataFrame
    variables: string, int or list of (strings or int).

    Returns
    -------
    filtered_variables: List of variables present in `variables` and in the
        input dataframe.
    """
    # When variables is not defined, keep it like this and return None.
    if variables is None:
        return None

    # If an integer or a string is provided, convert to a list.
    if not isinstance(variables, list):
        variables = [variables]

    # Filter out elements of variables that are not in the dataframe.
    filtered_variables = [var for var in variables if var in X.columns]

    # Raise an error if no column is left to work with.
    if len(filtered_variables) == 0:
        raise ValueError(
            "After filtering no variable remaining. At least 1 is required."
        )

    return filtered_variables


def find_categorical_and_numerical_variables(
    X: pd.DataFrame, variables: Variables = None
) -> Tuple[List[Union[str, int]], List[Union[str, int]]]:
    """
    Find numerical and categorical variables.

    The function returns two lists; the first one with the names of the variables of
    type object or categorical and the second list with the names of the numerical
    variables.

    More details in the :ref:`User Guide <find_cat_and_num_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list, default=None
        If `None`, the function will find categorical and numerical variables in X.
        Alternatively, it will find categorical and numerical variables in the given
        list.

    Returns
    -------
    variables: tuple
        List of numerical and list of categorical variables.

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

    # If the user passes just 1 variable outside a list.
    if isinstance(variables, (str, int)):

        if is_categorical(X[variables]) or is_object(X[variables]):
            variables_cat = [variables]
            variables_num = []
        elif is_numeric(X[variables]):
            variables_num = [variables]
            variables_cat = []
        else:
            raise TypeError(
                "The variable entered is neither numerical nor categorical."
            )

    # If user leaves default None parameter.
    elif variables is None:
        # find categorical variables
        if variables is None:
            variables_cat = [
                column
                for column in X.select_dtypes(include=["O", "category"]).columns
                if _is_categorical_and_is_not_datetime(X[column])
            ]
        # find numerical variables in dataset
        variables_num = list(X.select_dtypes(include="number").columns)

        if len(variables_num) == 0 and len(variables_cat) == 0:
            raise TypeError(
                "There are no numerical or categorical variables in the dataframe"
            )

    # If user passes variable list.
    else:
        if len(variables) == 0:
            raise ValueError("The list of variables is empty.")

        # find categorical variables
        variables_cat = [
            var for var in X[variables].select_dtypes(include=["O", "category"]).columns
        ]

        # find numerical variables
        variables_num = list(X[variables].select_dtypes(include="number").columns)

        if any([v for v in variables if v not in variables_cat + variables_num]):
            raise TypeError(
                "Some of the variables are neither numerical nor categorical."
            )

    return variables_cat, variables_num
