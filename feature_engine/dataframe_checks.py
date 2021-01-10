"""Series of checks to be performed on dataframes used as inputs of methods fit() and
transform().

"""

from typing import List, Union

import pandas as pd


def _is_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the input is a DataFrame and then creates a copy.

    Parameters
    ----------
    X : pandas Dataframe. The one that will be checked and copied.

    Raises
    ------
    TypeError
        If the input is not the Pandas DataFrame

    Returns
    -------
    X : pandas Dataframe.
        A copy of original DataFrame. Important step not to transform the original
        dataset of the user, accidentally.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X is not a pandas dataframe. The dataset should be a "
                        "pandas dataframe.")

    return X.copy()


def _check_input_matches_training_df(X: pd.DataFrame, reference: int) -> None:
    """
    Checks that DataFrame to transform has the same number of columns that the
    DataFrame used with the fit() method.

    Parameters
    ----------
    X : Pandas DataFrame
        The df to be checked
    reference : int
        The number of columns in the dataframe that was used with the fit() method.

    Raises
    ------
    ValueError
        If the number of columns does not match.

    Returns
    -------
    None
    """

    if X.shape[1] != reference:
        raise ValueError(
            "The number of columns in this dataset is different from the one used to "
            "fit this transformer (when using the fit() method)."
        )

    return None


def _check_contains_na(X: pd.DataFrame, variables: List[Union[str, int]]) -> None:
    """
    Checks if DataFrame contains null values in the selected columns.

    Parameters
    ----------
    X : Pandas DataFrame
    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain null values
    """

    if X[variables].isnull().values.any():
        raise ValueError(
            "Some of the variables to transform contain missing values. Check and "
            "remove those before using this transformer."
        )
