# Authors: Pradumna Suryawanshi <pradumnasuryawanshi@gmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class DropMissingData(BaseImputer):
    """
    The DropMissingData() will delete rows containing NA values.

    DropMissingData() will delete rows containing NA values from the variables
    indicated by the user, or variables with NA values in the train set.

    The DropMissingData() works for both numerical and categorical variables.
    The user can pass a list with the variables rows containing with NA should be dropped. Alternatively, the imputer will select variables with NA values
    from the training set.

    Parameters
    ----------
    missing_only : bool, defatult=True
        Indicates if rows with NA  should be dropped to variables with missing
        data or to all variables.

        True: rows with NA will be dropped only for those variables that showed
        missing data during fit.

        False: rows with NA will be dropped for all variables

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all variables with missing data.

    **Note**
    The transformer will first select all variables or all user entered
    variables and if how=missing_only, it will re-select from the original group
    only those that show missing data in during fit.

    Attributes
    ----------
    variables_:
        List of variables for which the rows with NA will be deleted.

    Methods
    -------
    fit:
        Learn the variables for which the rows with NA will be deleted
    transform:
        Add the missing indicators.
    fit_transform:
        Fit to the data, then trasnform it.
    return_dropped_data:
        Returns the dataframe with rows containing NA values.
    """

    def __init__(
        self,
        missing_only: bool = True,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if not isinstance(missing_only, bool):
            raise ValueError("missing_only takes values True or False")

        self.variables = _check_input_parameter_variables(variables)
        self.missing_only = missing_only

    def return_dropped_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the subset of the dataframe which contains rows with NA values.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The dataset to from which rows containing NA should be dropped.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame

        Returns
        -------
        X : pandas dataframe.
            The cleaned version of dataset that does not contain rows with NA values.
        """

        X = self._check_transform_input_and_state(X)

        idx = pd.isnull(X[self.variables_]).any(1)
        idx = idx[idx == True]
        return X.loc[idx.index, :]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the variables for which the rows with NA will be deleted.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y : pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find variables for which indicator should be added
        if self.missing_only:
            if not self.variables:
                self.variables_ = [
                    var for var in X.columns if X[var].isnull().sum() > 0
                ]
            else:
                self.variables_ = [
                    var for var in self.variables if X[var].isnull().sum() > 0
                ]

        else:
            if not self.variables:
                self.variables_ = [var for var in X.columns]
            else:
                self.variables_ = self.variables

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with NA values.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing rows with no NA values.
        """

        X = self._check_transform_input_and_state(X)

        X.dropna(axis=0, how="any", subset=self.variables, inplace=True)

        return X
