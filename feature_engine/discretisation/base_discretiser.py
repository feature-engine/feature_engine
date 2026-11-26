# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import Any, Dict, List

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine.dataframe_helpers import replace_columns_in_df, to_numpy


class BaseDiscretiser(BaseNumericalTransformer):
    """
    Shared set-up checks and methods across numerical discretisers.

    Important: inherits _fit_setup() and tags from BaseNumericalTransformer.
    Subclasses implement fit() themselves.
    """

    def __init__(
        self,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
    ) -> None:

        if not isinstance(return_object, bool):
            raise ValueError(
                "return_object must be True or False. " f"Got {return_object} instead."
            )

        if not isinstance(return_boundaries, bool):
            raise ValueError(
                "return_boundaries must be True or False. "
                f"Got {return_boundaries} instead."
            )

        if not isinstance(precision, int) or precision < 1:
            raise ValueError(
                "precision must be a positive integer. " f"Got {precision} instead."
            )

        self.return_object = return_object
        self.return_boundaries = return_boundaries
        self.precision = precision

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """Sort the variable values into the intervals.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """

        X = self._check_transform_input_and_state(X)
        arrays = to_numpy(X, self.variables_)
        X = replace_columns_in_df(X, self._discretise(arrays))
        return nw.to_native(X, pass_through=True)

    def _discretise(self, arrays: List[np.ndarray]) -> Dict[Any, np.ndarray]:
        """Sort the values of each variable into its intervals."""
        if self.return_boundaries is True:
            return {
                var: self._bin_labels(values, self.binner_dict_[var], self.precision)
                for var, values in zip(self.variables_, arrays)
            }
        return {
            var: self._bin_codes(values, self.binner_dict_[var], self.return_object)
            for var, values in zip(self.variables_, arrays)
        }

    def _digitize(self, values: np.ndarray, bins_arr: np.ndarray):
        """0-based bin index per value, right-closed intervals with the lowest edge
        included - mirrors pandas.cut(bins=bins, include_lowest=True), which is
        itself built on this same bins.searchsorted() call. Values outside the
        bin range, and NaNs, are flagged via na_mask rather than given a code.
        """
        ids = np.asarray(np.searchsorted(bins_arr, values, side="left"))
        ids[values == bins_arr[0]] = 1
        na_mask: np.ndarray = np.isnan(values) | (ids == len(bins_arr)) | (ids == 0)
        return ids - 1, na_mask

    def _bin_codes(self, values: np.ndarray, bins: List[float], return_object: bool):
        bins_arr: np.ndarray = np.asarray(bins, dtype=float)
        codes, na_mask = self._digitize(values, bins_arr)

        # match pandas.cut(labels=False): int codes, upcast to float only when a
        # NaN placeholder is actually needed.
        if na_mask.any():
            codes = codes.astype(np.float64)
            codes[na_mask] = np.nan
        if return_object is True:
            codes = codes.astype(object)

        return codes

    def _bin_labels(self, values: np.ndarray, bins: List[float], precision: int):
        bins_arr: np.ndarray = np.asarray(bins, dtype=float)
        codes, na_mask = self._digitize(values, bins_arr)

        labels = np.asarray(self._format_bin_labels(bins_arr, precision), dtype=object)
        out: np.ndarray = np.empty(len(values), dtype=object)
        out[~na_mask] = labels[codes[~na_mask]]
        out[na_mask] = None

        return out

    def _format_bin_labels(self, bins_arr: np.ndarray, precision: int) -> List[str]:
        """"(lower, upper]" text per bin, replicating pandas.cut's own label
        formatting: widen precision until break values are unique, then shrink
        the lowest edge so include_lowest values still read as inside the first
        interval.
        """
        precision = self._infer_precision(precision, bins_arr)
        breaks = [self._round_frac(b, precision) for b in bins_arr]
        breaks[0] = breaks[0] - 10 ** (-precision)
        return [f"({breaks[i]}, {breaks[i + 1]}]" for i in range(len(breaks) - 1)]

    def _round_frac(self, x: float, precision: int) -> float:
        if not np.isfinite(x) or x == 0:
            return float(x)
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            digits = precision
        return float(np.around(x, digits))

    def _infer_precision(self, base_precision: int, bins_arr: np.ndarray) -> int:
        # widen precision until every rounded break is unique - otherwise two
        # adjacent bins could render with identical label text.
        for precision in range(base_precision, 20):
            levels = [self._round_frac(b, precision) for b in bins_arr]
            if len(set(levels)) == len(bins_arr):
                return precision
        return base_precision
