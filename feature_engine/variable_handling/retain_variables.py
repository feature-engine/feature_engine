"""Functions to remove variables from a list."""

from typing import List, Union

Variables = Union[int, str, List[Union[str, int]]]


def retain_variables_if_in_df(X, variables):
    """Returns the subset of variables in the list that are present in the dataframe.

    More details in the :ref:`User Guide <retain_vars>`.

    Parameters
    ----------
    X:  pandas dataframe of shape = [n_samples, n_features]
        The dataset.

    variables: string, int or list of strings or int.
        The names of the variables to check.

    Returns
    -------
    variables_in_df: List.
        The subset of `variables` that is present `X`.

        Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import retain_variables_if_in_df
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> vars_in_df = retain_variables_if_in_df(X, ['var_num', 'var_cat', 'var_other'])
    >>> vars_in_df
    ['var_num', 'var_cat']
    """
    if isinstance(variables, (str, int)):
        variables = [variables]

    variables_in_df = [var for var in variables if var in X.columns]

    # Raise an error if no column is left to work with.
    if len(variables_in_df) == 0:
        raise ValueError(
            "None of the variables in the list are present in the dataframe."
        )

    return variables_in_df
