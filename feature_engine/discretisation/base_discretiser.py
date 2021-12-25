# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

import warnings
from typing import List, Optional, Union

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)


class BaseDiscretiser(BaseNumericalTransformer):
    """
    Share set-up checks and methods across numerical discretizers

    Parameters
    ----------
    return_object: bool, default=False
        Whether the the discrete variable should be returned as numeric or as
        object. If you would like to proceed with the engineering of the variable as if
        it was categorical, use True. Alternatively, keep the default to False.

    return_boundaries : bool, default=False
        Whether the output should be the interval boundaries. If True, it returns
        the interval boundaries. If False, it returns integers.


    """

    def __init__(
            self,
            return_object: bool = False,
            return_boundaries: bool = False,
            errors: str = "ignore",
    ) -> None:

        if not isinstance(return_object, bool):
            raise ValueError("return_object must be True or False. "
                             f"Got {return_object} instead.")

        if not isinstance(return_boundaries, bool):
            raise ValueError("return_boundaries must be True or False. "
                             f"Got {return_boundaries} instead.")

        if errors not in ["ignore", "raise"]:
            raise ValueError("errors only takes values 'ignore' and 'raise'. "
                             f"Got {errors} instead.")

        self.return_object = return_object
        self.return_boundaries = return_boundaries
        self.errors = errors

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X: Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values.
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X: Pandas DataFrame
            The same dataframe entered by the user.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check input data contains same number of columns as df used to fit
        _check_input_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        _check_contains_na(X, self.variables_)

        return X


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace categories with the learned parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The dataset to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing the categories replaced by numbers.
        """

        X = self._check_transform_input_and_state(X)

        # check if NaN values were introduced by the encoding
        if X[self.encoder_dict_.keys()].isnull().sum().sum() > 0:
            # obtain the name(s) of the columns have null values
            nan_columns = X.columns[X.isnull().any()].tolist()
            if len(nan_columns) > 1:
                nan_columns_str = ", ".join(nan_columns)
            else:
                nan_columns_str = nan_columns[0]

            warnings.warn(
                f"During the discretisation, NaN values were introduced in the feature(s) "
                f"{nan_columns_str}."
