import pandas as pd
import pytest

from feature_engine.datetime import DatetimeFeatures


@pytest.mark.parametrize("variables", ["date", "index"])
@pytest.mark.parametrize("timezone", [None, "Europe/London"])
def test_semi_month_boundaries(variables, timezone):
    dates = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-01-14",
            "2024-01-15",
            "2024-01-16",
            "2024-02-15",
            "2024-02-16",
            "2024-02-29",
            "2024-03-01",
            "2024-12-14",
            "2024-12-15",
            "2024-12-31",
            "2025-01-01",
        ],
        tz=timezone,
    )
    X = (
        pd.DataFrame({"date": dates}, index=range(20, 32))
        if variables == "date"
        else pd.DataFrame({"value": range(len(dates))}, index=dates)
    )
    original = X.copy(deep=True)
    transformer = DatetimeFeatures(
        variables=variables,
        features_to_extract=["semi_month_start", "semi_month_end"],
    )
    result = transformer.fit_transform(X)
    prefix = "date_" if variables == "date" else ""
    assert result[prefix + "semi_month_start"].tolist() == [
        1,
        1,
        2,
        2,
        4,
        4,
        4,
        5,
        23,
        24,
        24,
        1,
    ]
    assert result[prefix + "semi_month_end"].tolist() == [
        1,
        1,
        1,
        2,
        3,
        4,
        4,
        5,
        23,
        23,
        24,
        1,
    ]
    assert list(result.columns) == transformer.get_feature_names_out()
    pd.testing.assert_index_equal(result.index, X.index)
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("variables", ["date", "index"])
def test_semi_month_missing_dates(variables):
    dates = pd.DatetimeIndex(["2024-02-15", None, "2024-12-31"])
    X = (
        pd.DataFrame({"date": dates})
        if variables == "date"
        else pd.DataFrame({"value": [1, 2, 3]}, index=dates)
    )
    transformer = DatetimeFeatures(
        variables=variables,
        features_to_extract=["semi_month_start", "semi_month_end"],
        missing_values="ignore",
    )
    result = transformer.fit_transform(X)
    prefix = "date_" if variables == "date" else ""
    for feature, expected in [("semi_month_start", 4), ("semi_month_end", 3)]:
        column = result[prefix + feature]
        assert column.iloc[0] == expected
        assert pd.isna(column.iloc[1])
        assert column.iloc[2] == 24


def test_semi_month_all_months_and_default_features():
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    X = pd.DataFrame({"date": dates})
    result = DatetimeFeatures(features_to_extract="all").fit_transform(X)
    for feature in ["semi_month_start", "semi_month_end"]:
        assert result["date_" + feature].tolist() == list(range(1, 24, 2))
    default = DatetimeFeatures().fit_transform(X)
    assert not any("semi_month" in column for column in default.columns)
