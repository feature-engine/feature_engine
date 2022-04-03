"""Series of checks to be performed on dataframes used as inputs of methods fit() and
transform().
"""

from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .numpy_to_pandas import _is_numpy, _numpy_to_dataframe, _numpy_to_series


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
    if _is_numpy(X):
        X = _numpy_to_dataframe(X)

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


def _check_pd_X_y(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
):
    """
    Returns X as a DataFrame and y as a Series, converting any numpy
    objects to pandas objects as needed.
    * If both parameters are numpy objects, they are converted to pandas objects.
    * If one parameter is a pandas object and the other is a numpy object,
    the former will be converted to a pandas object, with the indexes
    of the latter.
    * If both parameters are pandas objects, and their indexes are inconsistent,
    an exception is raised (i.e. this is the caller's error.)
    * If both parameters are pandas objects and their indexes match, they are
    returned unchanged.
    * If X is sparse or X is empty or, after all transforms, is stiil
    not a DataFrame, raises an exception

    Parameters
    ----------
    X: Pandas DataFrame or numpy ndarray
    y: Pandas Series or numpy ndarray

    Returns
    -------
    X: Pandas DataFrame
    y: Pandas Series

    Exceptions
    ----------
    ValueError: if X and y are dimension-incompatible, X and y are pandas objects
    with inconsistent indexes
    """

    # * If both parameters are numpy objects, they are converted to pandas objects.
    # * If one parameter is a pandas object and the other is a numpy object,
    # the former will be converted to a pandas object, with the indexes
    # of the latter.
    if _is_numpy(X):
        X = _numpy_to_dataframe(X, index=y.index if isinstance(y, pd.Series) else None)
    if _is_numpy(y):
        y = _numpy_to_series(y, index=X.index if isinstance(X, pd.DataFrame) else None)

    # * If both parameters are pandas objects, and their indexes are inconsistent,
    # an exception is raised (i.e. this is the caller's error.)
    # * If both parameters are pandas objects and their indexes match, they are
    # returned unchanged.
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        if not all(y.index == X.index):
            raise ValueError("Index mismatch between DataFrame X and Series y")
        else:
            pass  # deliberately highlighting the no-op case

    # * If X is sparse or X is empty or, after all transforms, is stiil
    # not a DataFrame, raises an exception
    # (This deliberately carries out similar tests in _is_dataframe() above in
    # order to support different code paths)
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

    return X, y
