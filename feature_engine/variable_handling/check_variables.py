from typing import List, Union

import pandas as pd

Variables = Union[int, str, List[Union[str, int]]]


def check_numerical_variables(
    X: pd.DataFrame, variables: Variables
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are of type numerical.

    More details in the :ref:`User Guide <check_num_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : List
        List with the names of the variables to check.

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
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = check_numerical_variables(X, variables=["var_num"])
    >>> var_
    ['var_num']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    if len(X[variables].select_dtypes(exclude="number").columns) > 0:
        raise TypeError(
            "Some of the variables are not numerical. Please cast them as "
            "numerical before using this transformer."
        )

    return variables


def check_categorical_variables(
    X: pd.DataFrame, variables: Variables
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are of type object or categorical.

    More details in the :ref:`User Guide <check_cat_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list
        List with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the categorical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_categorical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = check_categorical_variables(X, "var_cat")
    >>> var_
    ['var_cat']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    if len(X[variables].select_dtypes(exclude=["O", "category"]).columns) > 0:
        raise TypeError(
            "Some of the variables are not categorical. Please cast them as "
            "object or categorical before using this transformer."
        )

    return variables
