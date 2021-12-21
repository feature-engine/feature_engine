# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.outliers.base_outlier import WinsorizerBase


class Winsorizer(WinsorizerBase):
    """
    The Winsorizer() caps maximum and/or minimum values of a variable at automatically
    determined values, and optionally adds indicators.

    The values to cap variables are determined using:

    - a Gaussian approximation
    - the inter-quantile range proximity rule (IQR)
    - percentiles

    **Gaussian limits:**

    - right tail: mean + 3* std
    - left tail: mean - 3* std

    **IQR limits:**

    - right tail: 75th quantile + 3* IQR
    - left tail:  25th quantile - 3* IQR

    where IQR is the inter-quartile range: 75th quantile - 25th quantile.

    **percentiles:**

    - right tail: 95th percentile
    - left tail:  5th percentile

    You can select how far out to cap the maximum or minimum values with the
    parameter `'fold'`.

    If `capping_method='gaussian'` fold gives the value to multiply the std.

    If `capping_method='iqr'` fold is the value to multiply the IQR.

    If `capping_method='quantiles'`, fold is the percentile on each tail that should
    be censored. For example, if fold=0.05, the limits will be the 5th and 95th
    percentiles. If fold=0.1, the limits will be the 10th and 90th percentiles.

    The Winsorizer() works only with numerical variables. A list of variables can
    be indicated. Alternatively, the Winsorizer() will select and cap all numerical
    variables in the train set.

    The transformer first finds the values at one or both tails of the distributions
    (fit). The transformer then caps the variables (transform).

    More details in the :ref:`User Guide <winsorizer>`.

    Parameters
    ----------
    capping_method: str, default='gaussian'
        Desired capping method. Can take 'gaussian', 'iqr' or 'quantiles'.

        **'gaussian'**: the transformer will find the maximum and / or minimum values
        to cap the variables using the Gaussian approximation.

        **'iqr'**: the transformer will find the boundaries using the IQR proximity
        rule.

        **'quantiles'**: the limits are given by the percentiles.

    tail: str, default='right'
        Whether to cap outliers on the right, left or both tails of the distribution.
        Can take 'left', 'right' or 'both'.

    fold: int or float, default=3
        How far out to to place the capping values. The number that will multiply
        the std or IQR to calculate the capping values. Recommended values, 2
        or 3 for the gaussian approximation, or 1.5 or 3 for the IQR proximity
        rule.

        If `capping_method='quantiles'`, then `'fold'` indicates the percentile. So if
        `fold=0.05`, the limits will be the 95th and 5th percentiles.

        **Note**: Outliers will be removed up to a maximum of the 20th percentiles on
        both sides. Thus, when `capping_method='quantiles'`, then `'fold'` takes values
        between 0 and 0.20.

    add_indicators: bool, default=False
        Whether to add indicator variables to flag the capped outliers.
        If 'True', binary variables will be added to flag outliers on the left and right
        tails of the distribution. One binary variable per tail, per variable.

    variables: list, default=None
        The list of variables for which the outliers will be capped. If None,
        the transformer will select and cap all numerical variables.

    missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. Sometimes we want to
        remove outliers in the raw, original data, sometimes, we may want to remove
        outliers in the already pre-transformed data. If `missing_values='ignore'`, the
        transformer will ignore missing data when learning the capping parameters or
        transforming the data. If `missing_values='raise'` the transformer will return
        an error if the training or the datasets to transform contain missing values.

    Attributes
    ----------
    right_tail_caps_:
        Dictionary with the maximum values at which variables will be capped.

    left_tail_caps_ :
        Dictionary with the minimum values at which variables will be capped.

    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Learn the values that should be used to replace outliers.
    transform:
        Cap the variables.
    fit_transform:
        Fit to the data. Then transform it.
    """

    def __init__(
        self,
        capping_method: str = "gaussian",
        tail: str = "right",
        fold: Union[int, float] = 3,
        add_indicators: bool = False,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
    ) -> None:
        if not isinstance(add_indicators, bool):
            raise ValueError(
                "add_indicators takes only booleans True and False"
                f"Got {add_indicators} instead."
            )
        super().__init__(capping_method, tail, fold, variables, missing_values)
        self.add_indicators = add_indicators

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cap the variable values. Optionally, add outlier indicators.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features + n_ind]
            The dataframe with the capped variables and indicators.
            The number of output variables depends on the values for 'tail' and
            'add_indicators': if passing 'add_indicators=False', will be equal
            to 'n_features', otherwise, will have an additional indicator column
            per processed feature for each tail.
        """
        if not self.add_indicators:
            return super().transform(X)
        else:
            X_orig = _is_dataframe(X)
            X_out = super().transform(X_orig)
            X_orig = X_orig[self.variables_]
            X_out_filtered = X_out[self.variables_]

            if self.tail in ["left", "both"]:
                X_left = X_out_filtered > X_orig
                X_left.columns = [str(cl) + "_left" for cl in self.variables_]
            if self.tail in ["right", "both"]:
                X_right = X_out_filtered < X_orig
                X_right.columns = [str(cl) + "_right" for cl in self.variables_]
            if self.tail == "left":
                X_out = pd.concat([X_out, X_left], axis=1)
            elif self.tail == "right":
                X_out = pd.concat([X_out, X_right], axis=1)
            else:
                X_both = pd.concat([X_left, X_right], axis=1)
                X_both = X_both[[
                    cl1 for cl2 in zip(X_left.columns.values, X_right.columns.values)
                    for cl1 in cl2
                ]]
                X_out = pd.concat([X_out, X_both], axis=1)
        return X_out
