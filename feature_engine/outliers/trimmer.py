# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd

from feature_engine.outliers import Winsorizer


class OutlierTrimmer(Winsorizer):
    """
    The OutlierTrimmer() removes observations with outliers from the dataset.


    It works only with numerical variables. A list of variables can be indicated.
    Alternatively, the OutlierTrimmer() will select all numerical variables.

    The OutlierTrimmer() first calculates the maximum and /or minimum values
    beyond which a value will be considered an outlier, and thus removed.

    Limits are determined using:
    1) A Gaussian approximation
    2) The interquartile range proximity rule
    3) Percentiles

    Gaussian limits:

        right tail: mean + 3 * std
        left tail: mean - 3 * std

    IQR limits:

        right tail: 75th percentile + 3 * IQR
        left tail:  25th percentile - 3 * IQR

    where IQR is the inter-quartile range:
        (75th percentile - 25th percentile) or (3th quartile - 1st quartile)

    Percentiles or quantiles:

        right tail: 95th percentile
        left tail:  5th percentile

    Attributes:
        Inherits from Winsorizer class

    Methods:
        transform(): Apply the transformation to the data

    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes observations with outliers from the dataframe.

        Args:
            X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns:
            The DataFrame without outlier observations.

        """

        X = self._check_transform_input_and_state(X)

        for feature in self.right_tail_caps_.keys():
            outliers = np.where(
                X[feature] > self.right_tail_caps_[feature], True, False
            )
            X = X.loc[~outliers]

        for feature in self.left_tail_caps_.keys():
            outliers = np.where(X[feature] < self.left_tail_caps_[feature], True, False)
            X = X.loc[~outliers]

        return X
