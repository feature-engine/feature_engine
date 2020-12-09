# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe, _check_contains_na
from feature_engine.outliers.base_outlier import BaseOutlier
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)


class Winsorizer(BaseOutlier):
    """
    The Winsorizer() caps maximum and / or minimum values of a variable.

    The Winsorizer() works only with numerical variables. A list of variables can
    be indicated. Alternatively, the Winsorizer() will select all numerical
    variables in the train set.

    The Winsorizer() first calculates the capping values at the end of the
    distribution. The values are determined using:

    - a Gaussian approximation,
    - the inter-quantile range proximity rule (IQR)
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
    (fit). The transformer then caps the variables (transform).

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

    variables: list, default=None
        The list of variables for which the outliers will be capped. If None,
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
        Dictionary with the maximum values at which variables will be capped.

    left_tail_caps_ :
        Dictionary with the minimum values at which variables will be capped.

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
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
    ) -> None:

        if capping_method not in ["gaussian", "iqr", "quantiles"]:
            raise ValueError(
                "capping_method takes only values 'gaussian', 'iqr' or 'quantiles'"
            )

        if tail not in ["right", "left", "both"]:
            raise ValueError("tail takes only values 'right', 'left' or 'both'")

        if fold <= 0:
            raise ValueError("fold takes only positive numbers")

        if capping_method == "quantiles" and fold > 0.2:
            raise ValueError(
                "with capping_method ='quantiles', fold takes values between 0 and "
                "0.20 only."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        self.capping_method = capping_method
        self.tail = tail
        self.fold = fold
        self.variables = _check_input_parameter_variables(variables)
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the values that should be used to replace outliers.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas Series, default=None
            y is not needed in this transformer. You can pass y or None.

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

        # find or check for numerical variables
        self.variables = _find_or_check_numerical_variables(X, self.variables)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}

        # estimate the end values
        if self.tail in ["right", "both"]:
            if self.capping_method == "gaussian":
                self.right_tail_caps_ = (
                    X[self.variables].mean() + self.fold * X[self.variables].std()
                ).to_dict()

            elif self.capping_method == "iqr":
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(
                    0.25
                )
                self.right_tail_caps_ = (
                    X[self.variables].quantile(0.75) + (IQR * self.fold)
                ).to_dict()

            elif self.capping_method == "quantiles":
                self.right_tail_caps_ = (
                    X[self.variables].quantile(1 - self.fold).to_dict()
                )

        if self.tail in ["left", "both"]:
            if self.capping_method == "gaussian":
                self.left_tail_caps_ = (
                    X[self.variables].mean() - self.fold * X[self.variables].std()
                ).to_dict()

            elif self.capping_method == "iqr":
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(
                    0.25
                )
                self.left_tail_caps_ = (
                    X[self.variables].quantile(0.25) - (IQR * self.fold)
                ).to_dict()

            elif self.capping_method == "quantiles":
                self.left_tail_caps_ = X[self.variables].quantile(self.fold).to_dict()

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseOutlier.transform.__doc__
