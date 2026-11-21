"""Functions used only by the estimators of the prediction module."""

from typing import Tuple

import narwhals as nw
import numpy as np


def target_mean_per_code(
    target: nw.Series, codes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the mean of the target per code, for the codes that have observations.

    Parameters
    ----------
    target : narwhals series
        The target, in the order of the rows of `codes`.

    codes : numpy array
        The integer code of each row, for example its bin.

    Returns
    -------
    codes_seen : numpy array
        The codes that have at least one observation, sorted.

    means : numpy array
        The mean of the target for each code in `codes_seen`.
    """
    # pandas is faster than narwhals.
    if target.implementation.is_pandas():
        means_per_code = target.to_native().groupby(codes).mean()
        return means_per_code.index.to_numpy(), means_per_code.to_numpy()
    else:
        stats = (
            target.to_frame()
            .with_columns(
                nw.new_series("__code__", codes, backend=target.implementation)
            )
            .group_by("__code__")
            .agg(nw.col(target.name).mean())
            .sort("__code__")
        )
        return stats["__code__"].to_numpy(), stats[target.name].to_numpy()
