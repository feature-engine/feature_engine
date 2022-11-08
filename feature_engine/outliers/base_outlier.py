from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._variable_handling.init_parameter_checks import (
    _check_init_parameter_variables,
)
from feature_engine._variable_handling.variable_type_selection import (
    _find_or_check_numerical_variables,
)
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine.tags import _return_tags


class BaseOutlier(BaseEstimator, TransformerMixin, GetFeatureNamesOutMixin):
    """shared set-up checks and methods across outlier transformers"""

    _right_tail_caps_docstring = """right_tail_caps_:
        Dictionary with the maximum values beyond which a value will be considered an
        outlier.
        """.rstrip()

    _left_tail_caps_docstring = """left_tail_caps_:
        Dictionary with the minimum values beyond which a value will be considered an
        outlier.
        """.rstrip()

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X: Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            If the dataframe is not of same size as that used in fit()

        Returns
        -------
        X: Pandas DataFrame
            The same dataframe entered by the user.
        """
        # check if class was fitted
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check that the dataframe contains the same number of columns
        # than the dataframe used to fit the imputer.
        _check_X_matches_training_df(X, self.n_features_in_)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        # reorder to match training set
        X = X[self.feature_names_in_]

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cap the variable values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """

        # check if class was fitted
        X = self._check_transform_input_and_state(X)

        # replace outliers
        for feature in self.right_tail_caps_.keys():
            X[feature] = X[feature].clip(upper=self.right_tail_caps_[feature])

        for feature in self.left_tail_caps_.keys():
            X[feature] = X[feature].clip(lower=self.left_tail_caps_[feature])

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        return tags_dict


class WinsorizerBase(BaseOutlier):

    _intro_docstring = """The extreme values beyond which an observation is considered
    an outlier are determined using:

    - a Gaussian approximation
    - the inter-quantile range proximity rule (IQR)
    - MAD-median rule (MAD)
    - percentiles

    **Gaussian limits:**

    - right tail: mean + 3* std
    - left tail: mean - 3* std

    **IQR limits:**

    - right tail: 75th quantile + 3* IQR
    - left tail:  25th quantile - 3* IQR

    where IQR is the inter-quartile range: 75th quantile - 25th quantile.

    **MAD limits:**

    - right tail: median + 3* MAD
    - left tail:  median - 3* MAD

    where MAD is the median absoulte deviation from the median.

    **percentiles:**

    - right tail: 95th percentile
    - left tail:  5th percentile

    You can select how far out to cap the maximum or minimum values with the
    parameter `'fold'`.

    If `capping_method='gaussian'` fold gives the value to multiply the std.

    If `capping_method='iqr'` fold is the value to multiply the IQR.

    If `capping_method='mad'` fold is the value to multiply the MAD.

    If `capping_method='quantiles'`, fold is the percentile on each tail that should
    be censored. For example, if fold=0.05, the limits will be the 5th and 95th
    percentiles. If fold=0.1, the limits will be the 10th and 90th percentiles.
    """.rstrip()

    _capping_method_docstring = """capping_method: str, default='gaussian'
        Desired outlier detection method. Can be 'gaussian', 'iqr', 'mad', 'quantiles'.

        The transformer will find the maximum and / or minimum values beyond which a
        data point will be considered an outlier using:
        **'gaussian'**: the Gaussian approximation.
        **'iqr'**: the IQR proximity rule.
        **'quantiles'**: the percentiles.
        **'mad'**: the Gaussian approximation but using robust statistics.
        """.rstrip()

    _tail_docstring = """tail: str, default='right'
        Whether to look for outliers on the right, left or both tails of the
        distribution. Can take 'left', 'right' or 'both'.
        """.rstrip()

    _fold_docstring = """fold: int or float, default=3
        The factor used to multiply the std, MAD or IQR to calculate
        the maximum or minimum allowed values.
        Recommended values are 2 or 3 for the gaussian approximation,
        1.5 or 3 for the IQR proximity rule and 3 or 3.5 for MAD rule.

        If `capping_method='quantile'`, then `'fold'` indicates the percentile. So if
        `fold=0.05`, the limits will be the 95th and 5th percentiles.

        **Note**: Outliers will be removed up to a maximum of the 20th percentiles on
        both sides. Thus, when `capping_method='quantile'`, then `'fold'` takes values
        between 0 and 0.20.
        """.rstrip()

    def __init__(
        self,
        capping_method: str = "gaussian",
        tail: str = "right",
        fold: Union[int, float] = 3,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
    ) -> None:

        if capping_method not in ["gaussian", "iqr", "quantiles", "mad"]:
            raise ValueError(
                "capping_method takes only values 'gaussian', 'iqr', 'mad', 'quantiles'"
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
        self.variables = _check_init_parameter_variables(variables)
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
        """

        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}

        if self.capping_method == "gaussian":
            bias = X[self.variables_].mean()
            scale = X[self.variables_].std(ddof=0)
        elif self.capping_method == "iqr":
            bias = X[self.variables_].quantile((0.75, 0.25))
            scale = bias.loc[0.75] - bias.loc[0.25]
        elif self.capping_method == "quantiles":
            bias = X[self.variables_].quantile((1 - self.fold, self.fold))
            scale = bias.loc[1 - self.fold] - bias.loc[self.fold]
        elif self.capping_method == "mad":
            bias = X[self.variables_].median()
            # scaling factor for normal distribution
            scale = (X[self.variables_] - bias).abs().median() / 0.67449
        if (scale == 0).any():
            raise ValueError(
                f"Input columns {scale[scale == 0].index.tolist()!r}"
                f" have low variation for method {self.capping_method!r}."
                f" Try other capping methods or drop these columns."
            )

        # estimate the end values
        if self.tail in ["right", "both"]:
            if self.capping_method in ("gaussian", "mad"):
                self.right_tail_caps_ = (bias + self.fold * scale).to_dict()

            elif self.capping_method == "iqr":
                self.right_tail_caps_ = (bias.loc[0.75] + self.fold * scale).to_dict()

            elif self.capping_method == "quantiles":
                self.right_tail_caps_ = bias.loc[1 - self.fold].to_dict()

        if self.tail in ["left", "both"]:
            if self.capping_method in ("gaussian", "mad"):
                self.left_tail_caps_ = (bias - self.fold * scale).to_dict()

            elif self.capping_method == "iqr":
                self.left_tail_caps_ = (bias.loc[0.25] - self.fold * scale).to_dict()

            elif self.capping_method == "quantiles":
                self.left_tail_caps_ = bias.loc[self.fold].to_dict()

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        # =======  this tests fail because the transformers throw an error
        # when variance of the any input feature is 0.
        # Nothing to do with the test itself but
        # mostly with the data created and used in the test
        msg = (
            "transformers raise errors when data variation is low, "
            "thus this check fails"
        )
        tags_dict["_xfail_checks"]["check_fit2d_1sample"] = msg
        return tags_dict
