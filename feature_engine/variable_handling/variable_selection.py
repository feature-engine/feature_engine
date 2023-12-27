"""Functions to select certain types of variables."""

from typing import List, Union

Variables = Union[None, int, str, List[Union[str, int]]]


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
