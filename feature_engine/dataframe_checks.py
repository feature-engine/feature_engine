"""Series of checks to be performed on dataframes used as inputs of methods fit() and
transform().
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils.validation import _check_y, check_consistent_length, column_or_1d


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
        if not X.columns.is_unique:
            raise ValueError("Input data contains duplicated variable names.")
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
        X.columns = [f"x{i}" for i in range(X.shape[1])]

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


def check_y(
    y: Union[np.generic, np.ndarray, pd.Series, pd.DataFrame, List],
    y_numeric: bool = False,
) -> pd.Series:
    """
    Checks that y is a series or a dataframe, or alternatively, if it can be converted
    to a series or dataframe.

    Parameters
    ----------
    y : pd.Series, pd.DataFrame, np.array, list
        The input to check and copy or transform.

    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    Returns
    -------
    y: pd.Series or pd.DataFrame
    """

    if y is None:
        raise ValueError(
            "requires y to be passed, but the target y is None",
            "Expected array-like (array or non-string sequence), got None",
            "y should be a 1d array",
        )

    elif isinstance(y, pd.Series):
        if y.isnull().any():
            raise ValueError("y contains NaN values.")
        if y.dtype != "O" and not np.isfinite(y).all():
            raise ValueError("y contains infinity values.")
        if y_numeric and y.dtype == "O":
            y = y.astype("float")
        y = y.copy()

    elif isinstance(y, pd.DataFrame):
        if y.isnull().any().any():
            raise ValueError("y contains NaN values.")
        if not np.isfinite(y).all().all():
            raise ValueError("y contains infinity values.")
        y = y.copy()

    else:
        try:
            y = column_or_1d(y)
            y = _check_y(y, multi_output=False, y_numeric=y_numeric)
            y = pd.Series(y).copy()
        except ValueError:
            y = _check_y(y, multi_output=True, y_numeric=y_numeric)
            y = pd.DataFrame(y).copy()
    return y


def check_X_y(
    X: Union[np.generic, np.ndarray, pd.DataFrame],
    y: Union[np.generic, np.ndarray, pd.Series, List],
    y_numeric: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Ensures X and y are compatible pandas DataFrame and Series. If both are pandas
    objects, checks that their indexes match. If any is a numpy array, converts to
    pandas object with compatible index.

    This transformer ensures that we can concatenate X and y using `pandas.concat`,
    functionality needed in the encoders.

    Parameters
    ----------
    X: Pandas DataFrame or numpy ndarray
        The input to check and copy or transform.

    y: pd.Series, np.array, list
        The input to check and copy or transform.

    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    Raises
    ------
    ValueError: if X and y are pandas objects with inconsistent indexes.
    TypeError: if X is sparse matrix, empty dataframe or not a dataframe.
    TypeError: if y can't be parsed as pandas Series.

    Returns
    -------
    X: Pandas DataFrame
    y: Pandas Series
    """

    def _check_X_y(X, y):
        X = check_X(X)
        y = check_y(y, y_numeric=y_numeric)
        check_consistent_length(X, y)
        return X, y

    # case 1: both are pandas objects
    if isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
        X, y = _check_X_y(X, y)
        # Check that their indexes match.
        if X.index.equals(y.index) is False:
            raise ValueError("The indexes of X and y do not match.")

    # case 2: X is dataframe and y is something else
    if isinstance(X, pd.DataFrame) and not isinstance(y, (pd.Series, pd.DataFrame)):
        X, y = _check_X_y(X, y)
        y.index = X.index

    # case 3: X is not a dataframe and y is a series
    elif not isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
        X, y = _check_X_y(X, y)
        X.index = y.index

    # all other cases
    else:
        X, y = _check_X_y(X, y)

    return X, y


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


def _check_contains_na(
    X: pd.DataFrame,
    variables: List[Union[str, int]],
) -> None:
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
        If the variable(s) contain null values.
    """

    if X[variables].isnull().any().any():
        raise ValueError(
            "Some of the variables in the dataset contain NaN. Check and "
            "remove those before using this transformer."
        )


def _check_optional_contains_na(
    X: pd.DataFrame, variables: List[Union[str, int]]
) -> None:
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
        If the variable(s) contain null values.
    """

    if X[variables].isnull().any().any():
        raise ValueError(
            "Some of the variables in the dataset contain NaN. Check and "
            "remove those before using this transformer or set the parameter "
            "`missing_values='ignore'` when initialising this transformer."
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

    if np.isinf(X[variables]).any().any():
        raise ValueError(
            "Some of the variables to transform contain inf values. Check and "
            "remove those before using this transformer."
        )
