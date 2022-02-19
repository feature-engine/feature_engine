"""Series of checks to be performed on dataframes used as inputs of methods fit() and
transform().
"""

from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse


def _is_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the input is a DataFrame and then creates a copy.

    If the input is a numpy array, it converts it to a pandas Dataframe. This is mostly
    so that we can add the check_estimator checks for compatibility with sklearn.

    Parameters
    ----------
    X : pandas Dataframe or numpy array. The one that will be checked and copied.

    Raises
    ------
    TypeError
        If the input is not a Pandas DataFrame or a numpy array

    Returns
    -------
    X : pandas Dataframe.
        A copy of original DataFrame. Important step not to accidentally transform the
        original dataset entered by the user.
    """
    # check_estimator uses numpy arrays for its checks.
    # Thus, we need to allow np arrays
    if isinstance(X, (np.generic, np.ndarray)):
        col_names = [str(i) for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=col_names)

    if issparse(X):
        raise ValueError("This transformer does not support sparse matrices.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X is not a pandas dataframe. The dataset should be a pandas dataframe."
        )

    if X.empty:
        raise ValueError(
            "0 feature(s) (shape=%s) while a minimum of %d is "
            "required." % (X.shape, 1)
        )

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


def _check_for_X_y_index_mismatch(
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray]
):
    """
    Handles the following case:
    1) X and y came from a DataFrame whose index was not the standard contiguous 0-n
    2) Earlier on, X was affected by an sklearn transform, turning it into an array and losing its different index
    3) X enters a feature-engine transformer and becomes a DataFrame again via _is_dataframe()
    4) X and y now have different indexes
    This function checks for this case; if it is met, it will return a copy of X with its index replaced with the
    correct index from y. It will not make any changes if either X or y is not a pandas object, or of course if there
    is no index mismatch.
    This case was first detected in issue #376.

    Parameters
    ----------
    X: Pandas DataFrame, Series, or numpy ndarray
        In all likelihood will be DataFrame, due to this method usually being called with the
        output of _is_dataframe(). Will not make any changes if X is not a DataFrame or Series.
    y: Pandas DataFrame, Series, or numpy ndarray
        In all likelihood will be a Series. Will not make any changes if X is not a DataFrame or Series.

    Returns
    -------
    X: same type as parameter X.
        If the mismatch conditions described above have been met, returns a copy of
        the original parameter, with index set to y's index. Else, returns original parameter value unaffected.
    """
    is_pd_X_and_y: bool = all([(type(i) == pd.DataFrame or type(i) == pd.Series) for i in (X, y)])
    if is_pd_X_and_y and any(X.index != y.index):
        X = X.copy()
        X.index = y.index

    return X