import pandas as pd
import pytest
from sklearn.base import clone

from feature_engine.timeseries.forecasting import (
    ExpandingWindowFeatures,
    LagFeatures,
    WindowFeatures,
)


def panel():
    return pd.DataFrame(
        {
            "store": [1, 2, 1, 2, 1, 2],
            "product": ["a"] * 6,
            "time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "sales": [10.0, 100.0, 20.0, 200.0, 30.0, 300.0],
        },
        index=[9, 7, 5, 3, 1, 0],
    )


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (LagFeatures, {"periods": [1, 2]}),
        (LagFeatures, {"freq": ["1D", "2D"]}),
        (
            WindowFeatures,
            {"window": [2, 3], "min_periods": 1, "functions": ["mean", "max"]},
        ),
        (WindowFeatures, {"window": "2D", "min_periods": 1}),
        (WindowFeatures, {"window": 2, "min_periods": 1, "freq": "1D"}),
        (ExpandingWindowFeatures, {"min_periods": 1}),
        (ExpandingWindowFeatures, {"min_periods": 1, "freq": "1D"}),
    ],
)
@pytest.mark.parametrize("drop_na", [False, True])
def test_grouped_output_matches_independent_series(cls, kwargs, drop_na):
    X = panel()
    original = X.copy(deep=True)
    transformer = cls(
        variables=["sales"],
        groupby=["store", "product"],
        sort_by="time",
        drop_na=drop_na,
        **kwargs,
    )
    actual = transformer.fit_transform(X)
    expected = []
    for _, group in X.groupby(["store", "product"], sort=False):
        subset = group.sort_values("time").set_index("time")
        result = cls(variables=["sales"], drop_na=drop_na, **kwargs).fit_transform(
            subset
        )
        result["time"] = result.index
        result.index = (
            group.set_index("time")
            .loc[result.index]
            .index.map(dict(zip(group.time, group.index)))
        )
        expected.append(result)
    expected = pd.concat(expected).reindex(actual.index)[actual.columns]
    expected.index.name = actual.index.name
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(X, original)
    assert list(actual.columns) == list(transformer.get_feature_names_out())
    clone(transformer).fit(X)


@pytest.mark.parametrize("cls", [LagFeatures, WindowFeatures, ExpandingWindowFeatures])
def test_group_columns_excluded_and_target_aligned(cls):
    X = panel()
    transformer = cls(groupby="store", sort_by="time", drop_na=True)
    transformer.fit(X)
    assert transformer.variables_ == ["sales"]
    actual, target = transformer.transform_x_y(X, pd.Series(range(6), index=X.index))
    pd.testing.assert_series_equal(
        target, pd.Series(range(6), index=X.index).loc[actual.index]
    )


def test_positional_lags_do_not_cross_groups():
    X = panel()
    result = LagFeatures(
        groupby="store", sort_index=False, fill_value=-1
    ).fit_transform(X)
    assert result.sales_lag_1.tolist() == [-1.0, -1.0, 10.0, 100.0, 20.0, 200.0]


def test_invalid_group_data():
    X = panel()
    with pytest.raises(ValueError, match="present"):
        LagFeatures(groupby="unknown").fit(X)
    with pytest.raises(ValueError, match="cannot be variables"):
        LagFeatures(groupby="store", variables="store").fit(X)
    X.loc[9, "store"] = float("nan")
    with pytest.raises(ValueError, match="missing"):
        LagFeatures(groupby="store").fit(X)


def test_combined_keys_define_groups_not_individual_columns():
    X = pd.DataFrame(
        {
            "store": [1, 1, 2, 2] * 2,
            "product": ["a", "b", "a", "b"] * 2,
            "sales": [1.0, 10.0, 100.0, 1000.0, 2.0, 20.0, 200.0, 2000.0],
        }
    )
    result = LagFeatures(groupby=["store", "product"], fill_value=-1).fit_transform(X)
    assert result.sales_lag_1.tolist() == [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        10.0,
        100.0,
        1000.0,
    ]


def test_grouped_window_drop_original_and_unseen_group():
    X = panel()
    transformer = WindowFeatures(
        groupby="store", sort_by="time", window=2, min_periods=1, drop_original=True
    ).fit(X)
    X["store"] += 100
    result = transformer.transform(X)
    assert "sales" not in result
    assert result.sales_window_2_mean.iloc[:2].isna().all()
    assert result.sales_window_2_mean.iloc[2:].tolist() == [10.0, 100.0, 15.0, 150.0]


def test_duplicate_time_within_group_is_rejected():
    X = panel()
    X.loc[5, "time"] = X.loc[9, "time"]
    with pytest.raises(ValueError, match="unique within each group"):
        LagFeatures(groupby="store", sort_by="time").fit(X)
