# Authors: Pradumna Suryawanshi <pradumnasuryawanshi@gmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class DropMissingData(BaseImputer):
    """
    The DropMissingData() will delete rows containing missing values. It provides
    similar functionality to pandas.drop_na().

    It works for both numerical and categorical variables. You can enter the list of
    variables for which missing values should be removed from the dataframe.
    Alternatively, the imputer will automatically select all variables in the dataframe.

    **Note**
    The transformer will first select all variables or all user entered
    variables and if `missing_only=True`, it will re-select from the original group
    only those that show missing data in during fit, that is in the train set.

    Parameters
    ----------
    missing_only: bool, default=True
        If true, missing observations will be dropped only for the variables that have
        missing data in the train set, during fit. If False, observations with NA
        will be dropped from all variables indicated by the user.

    variables: list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all variables in the dataframe.


    Attributes
    ----------
    variables_:
        List of variables for which the rows with NA will be deleted.
    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Learn the variables for which the rows with NA will be deleted
    transform:
        Remove observations with NA
    fit_transform:
        Fit to the data, then transform it.
    return_na_data:
        Returns the dataframe with the rows that contain NA .
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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the variables for which the rows with NA will be deleted.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
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

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_transformed: pandas dataframe
            The complete case dataframe for the selected variables, of shape
            [n_samples - rows_with_na, n_features]
        """

        X = self._check_transform_input_and_state(X)

        X.dropna(axis=0, how="any", subset=self.variables_, inplace=True)

        return X

    def return_na_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the subset of the dataframe which contains the rows with missing values.
        This method could be useful in production, in case we want to store the
        observations that will not be fed into the model.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame

        Returns
        -------
        X: pandas dataframe of shape = [obs_with_na, features]
            The dataframe containing only the rows with missing values.
        """

        X = self._check_transform_input_and_state(X)

        idx = pd.isnull(X[self.variables_]).any(1)
        idx = idx[idx]
        return X.loc[idx.index, :]
