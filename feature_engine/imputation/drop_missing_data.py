# Authors: Pradumna Suryawanshi <pradumnasuryawanshi@gmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class DropMissingData(BaseImputer):
    """
    DropMissingData() will delete rows containing missing values. It provides
    similar functionality to pandas.drop_na().

    It works for numerical and categorical variables. You can enter the list of
    variables for which missing values should be evaluated. Alternatively, the imputer
    will evaluate missing data in all variables in the dataframe.

    More details in the :ref:`User Guide <drop_missing_data>`.

    Parameters
    ----------
    missing_only: bool, default=True
        If `True`, rows will be dropped only if they show missing data in variables that
        have missing data in the train set, that is, the data set used in `fit()`. If
        `False`, rows will be dropped if there is missing data in any of the variables.

    variables: list, default=None
        The list of variables to consider for the imputation. If None, the imputer will
        evaluate missing data in all variables in the dataframe. Alternatively, the
        imputer will evaluate missing data only in the variables in the list.

        Note that if `missing_only=True` only variables with NA in the train set will
        be considered to drop a row, which might be a subset of your indicated list.

    tresh: float, default=None
        Require a certain percentage of missing data in a row to drop it. If `thresh=1`,
        all variables contemplated need to have NA to drop the row. If `thresh=0.5`,
        50% of the variables contemplated should show NA for a row to be dropped. If
        `thresh=None`, rows with NA in any of the variables will be dropped.

    Attributes
    ----------
    variables_:
        The variables for which missing data will be examined to decide if a row is
        dropped. The attribute `variables_` is different from the parameter `variables`
        when the latter is `None`, or when only a subset of the indicated variables
        show NA in the train set if `missing_only=True`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find the variables in which missing data should be evaluated.
    transform:
        Remove rows with NA.
    fit_transform:
        Fit to the data, then transform it.
    return_na_data:
        Returns the dataframe with the rows that contain NA.
    """

    def __init__(
        self,
        missing_only: bool = True,
        row_drop_pct: Optional[float] = None,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if not isinstance(missing_only, bool):
            raise ValueError("missing_only takes values True or False")

        if row_drop_pct:
            if not isinstance(row_drop_pct, float):
                raise TypeError(
                    f"row_drop_pct must be of type float. Got {row_drop_pct} instead."
                )
            if not 0.0 < row_drop_pct < 1.0:
                raise ValueError("row_drop_pct must be between 0.0 and 1.0")

        if missing_only & (row_drop_pct is not None):
            raise ValueError(
                f"If row_drop_pct is not None, missing_only must be set to False. \
                Got {missing_only} instead"
            )

        self.variables = _check_input_parameter_variables(variables)
        self.missing_only = missing_only
        self.row_drop_pct = row_drop_pct

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the variables for which the rows with NA will be deleted.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.

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
        X_new: pandas dataframe
            The complete case dataframe for the selected variables, of shape
            [n_samples - rows_with_na, n_features]
        """

        X = self._check_transform_input_and_state(X)

        if self.row_drop_pct:
            X.dropna(
                thresh=len(self.variables_) * (1 - self.row_drop_pct),
                subset=self.variables_,
                axis=0,
                inplace=True,
            )
        else:
            X.dropna(axis=0, how="any", subset=self.variables_, inplace=True)

        return X

    def return_na_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the subset of the dataframe which contains the rows with missing values.
        This method could be useful in production, in case we want to store the
        observations that will not be fed into the model.

        Parameters
        ----------
        X_na: pandas dataframe of shape = [obs_with_na, features]
            The dataframe to be transformed.
        """

        X = self._check_transform_input_and_state(X)

        idx = pd.isnull(X[self.variables_]).any(1)
        idx = idx[idx]
        return X.loc[idx.index, :]
