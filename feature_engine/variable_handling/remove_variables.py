"""Functions to remove variables from a list."""

from typing import List, Union

Variables = Union[int, str, List[Union[str, int]]]


def return_variables_if_in_df(X, variables):
    """Returns the subset of variables in the list that are present in the dataframe.

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
    """
    if isinstance(variables, (str, int)):
        variables = [variables]

    variables_in_df = [var for var in variables if var in X.columns]

    # Raise an error if no column is left to work with.
    if len(variables_in_df) == 0:
        raise ValueError(
            "None of the variables in the list is present in the dataframe."
        )

    return variables_in_df
