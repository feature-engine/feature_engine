from typing import Any, Dict, List, Literal, Optional, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.dataframe_helpers import replace_columns_in_df, subset_df, to_numpy
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


class BaseOutlier(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """shared set-up checks and methods across outlier transformers"""

    def _check_transform_input_and_state(
        self, X: IntoDataFrame
    ) -> Union[IntoDataFrame, nw.DataFrame]:
        """Checks that the input is a dataframe and of the same size as the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X: dataframe

        Raises
        ------
        TypeError
            If the input is not a recognised dataframe
        ValueError
            If the dataframe is not of same size as that used in fit()

        Returns
        -------
        X: pandas or narwhals dataframe
            The same pandas dataframe if X is pandas, or X in narwhals format
            otherwise, with the variables in the same order as in the train set.
        """
        check_is_fitted(self)
        X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)

        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        return subset_df(X, self.feature_names_in_)

    def _transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Cap the variable values.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """

        # check if class was fitted
        X = self._check_transform_input_and_state(X)
        capped = self._cap(dict(zip(self.variables_, to_numpy(X, self.variables_))))
        if len(capped) > 0:
            X = replace_columns_in_df(X, capped)

        return nw.to_native(X, pass_through=True)

    def _cap(self, arrays: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        """Returns the capped values of the variables that have a finite cap."""
        # infinite limits don't cap, and clipping to them turns integers into floats
        right = {v: c for v, c in self.right_tail_caps_.items() if np.isfinite(c)}
        left = {v: c for v, c in self.left_tail_caps_.items() if np.isfinite(c)}

        capped = {}
        for var, values in arrays.items():
            if var in left and var in right:
                capped[var] = np.clip(values, left[var], right[var])
            elif var in right:
                capped[var] = np.minimum(values, right[var])
            elif var in left:
                capped[var] = np.maximum(values, left[var])
        return capped

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags


class WinsorizerBase(BaseOutlier):

    _intro_docstring = """The extreme values beyond which an observation is considered
    an outlier are determined using:

    - a Gaussian approximation
    - the inter-quartile range proximity rule (IQR)
    - MAD-median rule (MAD)
    - percentiles

    **Gaussian limits:**

    - right tail: mean + 3* std
    - left tail: mean - 3* std

    **IQR limits:**

    - right tail: 75th quantile + 1.5* IQR
    - left tail:  25th quantile - 1.5* IQR

    where IQR is the inter-quartile range: 75th quantile - 25th quantile.

    **MAD limits:**

    - right tail: median + 3.29* MAD
    - left tail:  median - 3.29* MAD

    where MAD is the median absolute deviation from the median.

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

    def __init__(
        self,
        capping_method: str = "gaussian",
        tail: str = "right",
        fold: Union[int, float, Literal["auto"]] = "auto",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        missing_values: str = "raise",
    ) -> None:

        if capping_method not in ("gaussian", "iqr", "quantiles", "mad"):
            raise ValueError(
                f"capping_method must be 'gaussian', 'iqr', 'mad', 'quantiles'."
                f" Got {capping_method} instead."
            )

        if tail not in ("right", "left", "both"):
            raise ValueError(
                f"tail must be 'right', 'left' or 'both'. Got {tail} instead."
            )

        if (isinstance(fold, str) and (fold != "auto")) or (
            isinstance(fold, (int, float)) and (fold <= 0)
        ):
            raise ValueError(
                f"fold must be a positive number or 'auto'. Got {fold} instead."
            )

        if (
            capping_method == "quantiles"
            and isinstance(fold, (int, float))
            and fold > 0.2
        ):
            raise ValueError(
                "with capping_method ='quantiles', fold takes values between 0 and "
                "0.20 only."
            )

        if missing_values not in ("raise", "ignore"):
            raise ValueError(
                f"missing_values must be 'raise' or 'ignore'."
                f" Got {missing_values} instead."
            )

        _check_return_empty_is_bool(return_empty)

        self.capping_method = capping_method
        self.tail = tail
        self.fold = fold
        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty
        self.missing_values = missing_values

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Learn the values that should be used to replace outliers.

        Parameters
        ----------
        X : dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : Series, default=None
            y is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        if self.variables is None:
            self.variables_ = find_numerical_variables(
                X, return_empty=self.return_empty
            )
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}

        if self.fold == "auto":
            self.fold_ = self._calculate_fold()
        else:
            self.fold_ = self.fold

        arrays = to_numpy(X, self.variables_)
        if len(arrays) > 0:
            # transposing keeps each variable contiguous in memory, which makes the
            # reductions per variable faster than with np.column_stack().
            values = np.array(arrays).T
        else:
            values = np.empty((len(X), 0))

        # nan-aware reductions: with missing_values="ignore", values may contain
        # NaN, and pandas' mean/std/quantile/median skip NaN by default.
        if self.capping_method == "gaussian":
            bias = np.nanmean(values, axis=0)
            scale = np.nanstd(values, axis=0, ddof=0)
        elif self.capping_method == "iqr":
            # one call for both quantiles sorts each variable once
            q25, q75 = np.nanquantile(values, [0.25, 0.75], axis=0)
            scale = q75 - q25
        elif self.capping_method == "quantiles":
            q_lo, q_hi = np.nanquantile(values, [self.fold_, 1 - self.fold_], axis=0)
            scale = q_hi - q_lo
        elif self.capping_method == "mad":
            bias = np.nanmedian(values, axis=0)
            # scaling factor for normal distribution
            scale = np.nanmedian(np.abs(values - bias), axis=0) / 0.67449

        # estimate the end values
        if self.tail in ("right", "both"):
            if self.capping_method in ("gaussian", "mad"):
                self.right_tail_caps_ = {
                    var: float(b + self.fold_ * s)
                    for var, b, s in zip(self.variables_, bias, scale)
                }

            elif self.capping_method == "iqr":
                self.right_tail_caps_ = {
                    var: float(q + self.fold_ * s)
                    for var, q, s in zip(self.variables_, q75, scale)
                }

            elif self.capping_method == "quantiles":
                self.right_tail_caps_ = {
                    var: float(q) for var, q in zip(self.variables_, q_hi)
                }

        if self.tail in ("left", "both"):
            if self.capping_method in ("gaussian", "mad"):
                self.left_tail_caps_ = {
                    var: float(b - self.fold_ * s)
                    for var, b, s in zip(self.variables_, bias, scale)
                }

            elif self.capping_method == "iqr":
                self.left_tail_caps_ = {
                    var: float(q - self.fold_ * s)
                    for var, q, s in zip(self.variables_, q25, scale)
                }

            elif self.capping_method == "quantiles":
                self.left_tail_caps_ = {
                    var: float(q) for var, q in zip(self.variables_, q_lo)
                }

        # variables without variation have no outliers, so they get infinite limits
        for var, s in zip(self.variables_, scale):
            if s == 0:
                if var in self.right_tail_caps_:
                    self.right_tail_caps_[var] = float("inf")
                if var in self.left_tail_caps_:
                    self.left_tail_caps_[var] = float("-inf")

        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        return self

    def _calculate_fold(self) -> float:
        if self.capping_method == "quantiles":
            return 0.05
        elif self.capping_method == "iqr":
            return 1.5
        elif self.capping_method == "mad":
            return 3.29
        else:
            return 3.0

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        # =======  this tests fail because the transformers throw an error
        # when variance of any input feature is 0.
        # Nothing to do with the test itself but
        # mostly with the data created and used in the test
        msg = (
            "transformers raise errors when data variation is low, "
            "thus this check fails"
        )
        tags_dict["_xfail_checks"]["check_fit2d_1sample"] = msg
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
