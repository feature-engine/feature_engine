# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Union

import pandas as pd


def _is_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the input is a DataFrame.
    Creates the copy of the initial DataFrame.

    Args:
        X: Argument to perform check against

    Raises:
        TypeError: If the input is not the Pandas DataFrame

    Returns:
        The copy of initial DataFrame.
        Important not to transform the original dataset.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("The data set should be a pandas dataframe")

    return X.copy()


def _check_input_matches_training_df(X: pd.DataFrame, reference: int) -> None:
    """
    Check that DataFrame to transform has the same number
    of columns that the DataFrame used during fit method.

    Args:
        X: Pandas DataFrame to perform comparison
        reference: Number of columns

    Raises:
        ValueError: If number of columns is not the same

    Returns:
        None
    """

    if X.shape[1] != reference:
        raise ValueError(
            "The number of columns in this data set is different from the one used to "
            "fit this transformer (when using the fit method)"
        )

    return None


def _check_contains_na(X: pd.DataFrame, variables: Union[str, List[str]]):
    """
    Checks if DataFrame columns contain null values.

    Args:
        X: Pandas DataFrame to perform check against
        variables: List of variables to check for null values

    Raises:
        ValueError: If variable(s) contain null values
    """
    if X[variables].isnull().values.any():
        raise ValueError(
            "Some of the variables to transform contain missing values. Check and "
            "remove those before using this transformer."
        )
