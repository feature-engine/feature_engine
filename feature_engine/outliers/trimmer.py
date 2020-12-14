# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd

from feature_engine.outliers import Winsorizer


class OutlierTrimmer(Winsorizer):
    """The OutlierTrimmer() removes observations with outliers from the dataset.

    It works only with numerical variables. A list of variables can be indicated.
    Alternatively, the OutlierTrimmer() will select all numerical variables.

    The OutlierTrimmer() first calculates the maximum and /or minimum values
    beyond which a value will be considered an outlier, and thus removed.

    Limits are determined using:

    - a Gaussian approximation
    - the inter-quantile range proximity rule
    - percentiles.

    **Gaussian limits:**

    - right tail: mean + 3* std
    - left tail: mean - 3* std

    **IQR limits:**

    - right tail: 75th quantile + 3* IQR
    - left tail:  25th quantile - 3* IQR

    where IQR is the inter-quartile range: 75th quantile - 25th quantile.

    **percentiles or quantiles:**

    - right tail: 95th percentile
    - left tail:  5th percentile

    You can select how far out to cap the maximum or minimum values with the
    parameter 'fold'.

    If `capping_method='gaussian'` fold gives the value to multiply the std.

    If `capping_method='iqr'` fold is the value to multiply the IQR.

    If `capping_method='quantile'`, fold is the percentile on each tail that should
    be censored. For example, if fold=0.05, the limits will be the 5th and 95th
    percentiles. If fold=0.1, the limits will be the 10th and 90th percentiles.

    The transformer first finds the values at one or both tails of the distributions
    (fit).

    The transformer then removes observations with outliers from the dataframe
    (transform).

    Parameters
    ----------

    capping_method : str, default=gaussian
        Desired capping method. Can take 'gaussian', 'iqr' or 'quantiles'.

        'gaussian': the transformer will find the maximum and / or minimum values to
        cap the variables using the Gaussian approximation.

        'iqr': the transformer will find the boundaries using the IQR proximity rule.

        'quantiles': the limits are given by the percentiles.

    tail : str, default=right
        Whether to cap outliers on the right, left or both tails of the distribution.
        Can take 'left', 'right' or 'both'.

    fold: int or float, default=3
        How far out to to place the capping values. The number that will multiply
        the std or IQR to calculate the capping values. Recommended values, 2
        or 3 for the gaussian approximation, or 1.5 or 3 for the IQR proximity
        rule.

        If capping_method='quantile', then 'fold' indicates the percentile. So if
        fold=0.05, the limits will be the 95th and 5th percentiles.
        **Note**: Outliers will be removed up to a maximum of the 20th percentiles on
        both sides. Thus, when capping_method='quantile', then 'fold' takes values
        between 0 and 0.20.

    variables : list, default=None
        The list of variables for which the outliers will be removed If None,
        the transformer will find and select all numerical variables.

    missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. Sometimes we want to
        remove outliers in the raw, original data, sometimes, we may want to remove
        outliers in the already pre-transformed data. If missing_values='ignore', the
        transformer will ignore missing data when learning the capping parameters or
        transforming the data. If missing_values='raise' the transformer will return
        an error if the training or the datasets to transform contain missing values.

    Attributes
    ----------
    right_tail_caps_:
        Dictionary with the maximum values above which values will be removed

    left_tail_caps_ :
        Dictionary with the minimum values below which values will be removed

    Methods
    -------
    fit:
        Find maximum and minimum values.
    transform:
        Remove outliers.
    fit_transform:
        Fit to the data. Then transform it.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove observations with outliers from the dataframe.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            If the dataframe is not of same size as that used in fit()

        Returns
        -------
        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe without outlier observations.
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
