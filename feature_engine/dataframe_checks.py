"""Series of checks to be performed on dataframes used as inputs of methods fit() and
transform().
"""

from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse


def check_X(X: Union[np.generic, np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Checks if the input is a DataFrame and then creates a copy. This is an important
    step not to accidentally transform the original dataset entered by the user.

    If the input is a numpy array, it converts it to a pandas Dataframe. The column
    names are strings representing the column index starting at 0.

    Feature-engine was originally designed to work with pandas dataframes. However,
    allowing numpy arrays as input allows 2 things:

    We can use the Scikit-learn tests for transformers provided by the
    `check_estimator` function to test the compatibility of our transformers with
    sklearn functionality.

    Feature-engine transformers can be used within a Scikit-learn Pipeline together
    with Scikit-learn transformers like the `SimpleImputer`, which return by default
    Numpy arrays.

    Parameters
    ----------
    X : pandas Dataframe or numpy array.
        The input to check and copy or transform.

    Raises
    ------
    TypeError
        If the input is not a Pandas DataFrame or a numpy array.
    ValueError
        If the input is an empty dataframe.

    Returns
    -------
    X : pandas Dataframe.
        A copy of original DataFrame or a converted Numpy array.
    """
    if isinstance(X, pd.DataFrame):
        X = X.copy()

    elif isinstance(X, (np.generic, np.ndarray)):
        # If input is scalar raise error
        if X.ndim == 0:
            raise ValueError(
                "Expected 2D array, got scalar array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(X)
            )
        # If input is 1D raise error
        if X.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(X)
            )

        X = pd.DataFrame(X)
        X.columns = [str(i) for i in range(X.shape[1])]

    elif issparse(X):
        raise TypeError("This transformer does not support sparse matrices.")

    else:
        raise TypeError(
            f"X must be a numpy array or pandas dataframe. Got {type(X)} instead."
        )

    if X.empty:
        raise ValueError(
            "0 feature(s) (shape=%s) while a minimum of %d is required." % (X.shape, 1)
        )

    return X


def _check_X_matches_training_df(X: pd.DataFrame, reference: int) -> None:
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
            "Some of the variables to transform contain NaN. Check and "
            "remove those before using this transformer."
        )


def _check_contains_inf(X: pd.DataFrame, variables: List[Union[str, int]]) -> None:
    """
    Checks if DataFrame contains inf values in the selected columns.

    Parameters
    ----------
    X : Pandas DataFrame
    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain np.inf values
    """

    if np.isinf(X[variables]).values.any():
        raise ValueError(
            "Some of the variables to transform contain inf values. Check and "
            "remove those before using this transformer."
        )
