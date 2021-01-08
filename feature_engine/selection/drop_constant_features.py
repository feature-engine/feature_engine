from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_contains_na,
)
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
)
from feature_engine.selection.base_selector import BaseSelector

Variables = Union[None, int, str, List[Union[str, int]]]


class DropConstantFeatures(BaseSelector):
    """
    Drop constant and quasi-constant variables from a dataframe. Constant variables
    show the same value across all the observations in the dataset. Quasi-constant
    variables show the same value in almost all the observations in the dataset.

    By default, DropConstantFeatures() drops only constant variables. This transformer
    works with both numerical and categorical variables. The user can indicate a list
    of variables to examine. Alternatively, the transformer will evaluate all the
    variables in the dataset.

    The transformer will first identify and store the constant and quasi-constant
    variables. Next, the transformer will drop these variables from a dataframe.

    Parameters
    ----------
    tol : float,int,  default=1
        Threshold to detect constant/quasi-constant features. Variables showing the
        same value in a percentage of observations greater than tol will be considered
        constant / quasi-constant and dropped. If tol=1, the transformer removes
        constant variables. Else, it will remove quasi-constant variables.

    variables : list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset.

    missing_values : str, default=raises
        Whether the missing values should be raised as error, ignored or included as an
        additional value of the variable, when considering if the feature is constant
        or quasi-constant. Takes values 'raise', 'ignore', 'include'.

    Attributes
    ----------
    features_to_drop_:
        List with constant and quasi-constant features.

    Methods
    -------
    fit:
        Find constant and quasi-constant features.
    transform:
        Remove constant and quasi-constant features.
    fit_transform:
        Fit to the data. Then transform it.

    Notes
    -----
    This transformer is a similar concept to the VarianceThreshold from Scikit-learn,
    but it evaluates number of unique values instead of variance

    See Also
    --------
    sklearn.feature_selection.VarianceThreshold
    """

    def __init__(
        self, tol: float = 1, variables: Variables = None, missing_values: str = "raise"
    ):

        if not isinstance(tol, (float, int)) or tol < 0 or tol > 1:
            raise ValueError("tol must be a float or integer between 0 and 1")

        if missing_values not in ["raise", "ignore", "include"]:
            raise ValueError(
                "missing_values takes only values 'raise', 'ignore' or " "'include'."
            )

        self.tol = tol
        self.variables = _check_input_parameter_variables(variables)
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Find constant and quasi-constant features.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.
        y : None
            y is not needed for this transformer. You can pass y or None.

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are present in the dataframe
        self.variables = _find_all_variables(X, self.variables)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        if self.missing_values == "include":
            X[self.variables] = X[self.variables].fillna("missing_values")

        # find constant features
        if self.tol == 1:
            self.features_to_drop_ = [
                feature for feature in self.variables if X[feature].nunique() == 1
            ]

        # find constant and quasi-constant features
        else:
            self.features_to_drop_ = []

            for feature in self.variables:
                # find most frequent value / category in the variable
                predominant = (
                    (X[feature].value_counts() / np.float(len(X)))
                    .sort_values(ascending=False)
                    .values[0]
                )

                if predominant >= self.tol:
                    self.features_to_drop_.append(feature)

        # check we are not dropping all the columns in the df
        if len(self.features_to_drop_) == len(X.columns):
            raise ValueError(
                "The resulting dataframe will have no columns after dropping all "
                "constant or quasi-constant features. Try changing the tol value."
            )

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseSelector.transform.__doc__
