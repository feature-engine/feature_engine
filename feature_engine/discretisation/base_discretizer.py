# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.dataframe_checks import _check_contains_na


class BaseDiscretizer(BaseNumericalTransformer):
    """ Share set-up checks and methods across numerical discretizers"""

    def __init__(self):
        pass


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

        X = X._self

        # check if NaN values were introduced by the encoding
        if X[self.encoder_dict_.keys()].isnull().sum().sum() > 0:
            # obtain the name(s) of the columns have null values
            nan_columns = X.columns[X.isnull().any()].tolist()
            if len(nan_columns) > 1:
                nan_columns_str = ", ".join(nan_columns)
            else:
                nan_columns_str = nan_columns[0]

            warnings.warn(
                f"NaN values were introduced in the feature(s) "
                f"{nan_columns_str} during the encoding."
