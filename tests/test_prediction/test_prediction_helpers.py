import narwhals as nw
import numpy as np
import pandas as pd
import pytest

from feature_engine._prediction._prediction_helpers import target_mean_per_code
from tests.backend_helpers import make_series


def test_target_mean_per_code_returns_sorted_codes_with_observations(make_df):
    y = make_series(make_df, [1.0, 2.0, 3.0, 4.0, 6.0])
    target = nw.from_native(y, series_only=True)
    codes = np.array([3, 0, 3, 0, 1])

    codes_seen, means = target_mean_per_code(target, codes)

    assert codes_seen.tolist() == [0, 1, 3]
    assert means.tolist() == pytest.approx([3.0, 6.0, 2.0])


def test_target_mean_per_code_pairs_pandas_rows_by_position():
    y = pd.Series([1.0, 2.0, 4.0], index=[5, 0, 9])
    target = nw.from_native(y, series_only=True)

    codes_seen, means = target_mean_per_code(target, np.array([1, 0, 1]))

    assert codes_seen.tolist() == [0, 1]
    assert means.tolist() == pytest.approx([2.0, 2.5])
