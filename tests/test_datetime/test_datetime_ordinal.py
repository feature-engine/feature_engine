import datetime
import pandas as pd
import pytest

from feature_engine.datetime import DatetimeOrdinal


@pytest.fixture(scope="module")
def df_datetime_ordinal():
    df = pd.DataFrame({
        "date_col_1": pd.to_datetime(
            ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        ),
        "date_col_2": pd.to_datetime(
            ["2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14"]
        ),
        "non_date_col": [1, 2, 3, 4, 5],
    })
    return df


@pytest.fixture(scope="module")
def df_datetime_ordinal_na():
    df = pd.DataFrame({
        "date_col_1": pd.to_datetime(
            ["2023-01-01", "2023-01-02", None, "2023-01-04", "2023-01-05"]
        ),
        "date_col_2": pd.to_datetime(
            ["2024-02-10", "2024-02-11", "2024-02-12", None, "2024-02-14"]
        ),
    })
    return df


def test_datetime_ordinal_no_start_date(df_datetime_ordinal):
    transformer = DatetimeOrdinal(variables=["date_col_1", "date_col_2"])
    X_transformed = transformer.fit_transform(df_datetime_ordinal)

    expected_ordinal_1 = pd.Series(
        [d.toordinal() for d in df_datetime_ordinal["date_col_1"]],
        name="date_col_1_ordinal",
    )
    expected_ordinal_2 = pd.Series(
        [d.toordinal() for d in df_datetime_ordinal["date_col_2"]],
        name="date_col_2_ordinal",
    )

    pd.testing.assert_series_equal(
        X_transformed["date_col_1_ordinal"], expected_ordinal_1
    )
    pd.testing.assert_series_equal(
        X_transformed["date_col_2_ordinal"], expected_ordinal_2
    )
    assert "non_date_col" in X_transformed.columns
    assert "date_col_1" not in X_transformed.columns
    assert "date_col_2" not in X_transformed.columns


def test_datetime_ordinal_with_start_date(df_datetime_ordinal):
    start_date_str = "2023-01-01"
    transformer = DatetimeOrdinal(variables=["date_col_1"], start_date=start_date_str)
    X_transformed = transformer.fit_transform(df_datetime_ordinal)

    start_ordinal = pd.to_datetime(start_date_str).toordinal()
    expected_ordinal = pd.Series(
        [d.toordinal() - start_ordinal + 1 for d in df_datetime_ordinal["date_col_1"]],
        name="date_col_1_ordinal",
    )

    pd.testing.assert_series_equal(
        X_transformed["date_col_1_ordinal"], expected_ordinal
    )
    assert "date_col_2" in X_transformed.columns
    assert "date_col_1" not in X_transformed.columns


def test_datetime_ordinal_with_start_date_datetime_object(df_datetime_ordinal):
    start_date_obj = datetime.date(2023, 1, 1)
    transformer = DatetimeOrdinal(variables=["date_col_1"], start_date=start_date_obj)
    X_transformed = transformer.fit_transform(df_datetime_ordinal)

    start_ordinal = pd.to_datetime(start_date_obj).toordinal()
    expected_ordinal = pd.Series(
        [d.toordinal() - start_ordinal + 1 for d in df_datetime_ordinal["date_col_1"]],
        name="date_col_1_ordinal",
    )

    pd.testing.assert_series_equal(
        X_transformed["date_col_1_ordinal"], expected_ordinal
    )


def test_datetime_ordinal_no_variables_specified(df_datetime_ordinal):
    transformer = DatetimeOrdinal()
    X_transformed = transformer.fit_transform(df_datetime_ordinal)

    expected_ordinal_1 = pd.Series(
        [d.toordinal() for d in df_datetime_ordinal["date_col_1"]],
        name="date_col_1_ordinal",
    )
    expected_ordinal_2 = pd.Series(
        [d.toordinal() for d in df_datetime_ordinal["date_col_2"]],
        name="date_col_2_ordinal",
    )

    pd.testing.assert_series_equal(
        X_transformed["date_col_1_ordinal"], expected_ordinal_1
    )
    pd.testing.assert_series_equal(
        X_transformed["date_col_2_ordinal"], expected_ordinal_2
    )
    assert "non_date_col" in X_transformed.columns
    assert "date_col_1" not in X_transformed.columns
    assert "date_col_2" not in X_transformed.columns


def test_datetime_ordinal_missing_values_raise(df_datetime_ordinal_na):
    transformer = DatetimeOrdinal(missing_values="raise")
    with pytest.raises(ValueError):
        transformer.fit(df_datetime_ordinal_na)


def test_datetime_ordinal_missing_values_ignore(df_datetime_ordinal_na):
    transformer = DatetimeOrdinal(missing_values="ignore")
    X_transformed = transformer.fit_transform(df_datetime_ordinal_na)

    # Expected values for date_col_1_ordinal, handling None
    expected_ordinal_1 = pd.Series(
        [
            d.toordinal() if pd.notna(d) else pd.NA
            for d in df_datetime_ordinal_na["date_col_1"]
        ],
        name="date_col_1_ordinal",
        dtype=object,
    )
    expected_ordinal_2 = pd.Series(
        [
            d.toordinal() if pd.notna(d) else pd.NA
            for d in df_datetime_ordinal_na["date_col_2"]
        ],
        name="date_col_2_ordinal",
        dtype=object,
    )

    pd.testing.assert_series_equal(
        X_transformed["date_col_1_ordinal"], expected_ordinal_1
    )
    pd.testing.assert_series_equal(
        X_transformed["date_col_2_ordinal"], expected_ordinal_2
    )


def test_datetime_ordinal_invalid_start_date():
    with pytest.raises(ValueError):
        DatetimeOrdinal(start_date="not-a-date")


def test_datetime_ordinal_non_datetime_variable_error(df_datetime_ordinal):
    transformer = DatetimeOrdinal(variables=["non_date_col"])
    with pytest.raises(TypeError):
        transformer.fit(df_datetime_ordinal)


def test_datetime_ordinal_drop_original_false(df_datetime_ordinal):
    transformer = DatetimeOrdinal(variables=["date_col_1"], drop_original=False)
    X_transformed = transformer.fit_transform(df_datetime_ordinal)

    assert "date_col_1" in X_transformed.columns
    assert "date_col_1_ordinal" in X_transformed.columns
    assert "date_col_2" in X_transformed.columns


def test_datetime_ordinal_get_feature_names_out(df_datetime_ordinal):
    transformer = DatetimeOrdinal(variables=["date_col_1", "date_col_2"])
    transformer.fit(df_datetime_ordinal)
    feature_names_out = transformer.get_feature_names_out()

    expected_feature_names = [
        "date_col_1_ordinal",
        "date_col_2_ordinal",
        "non_date_col",
    ]
    assert sorted(feature_names_out) == sorted(expected_feature_names)


def test_datetime_ordinal_get_feature_names_out_with_input_features(
    df_datetime_ordinal,
):
    transformer = DatetimeOrdinal(variables=["date_col_1"], drop_original=False)
    transformer.fit(df_datetime_ordinal)
    feature_names_out = transformer.get_feature_names_out(
        input_features=df_datetime_ordinal.columns.tolist()
    )

    expected_feature_names = [
        "date_col_1_ordinal",
        "date_col_2",
        "non_date_col",
        "date_col_1",
    ]
    assert sorted(feature_names_out) == sorted(expected_feature_names)


def test_datetime_ordinal_get_feature_names_out_with_input_features_drop_original(
    df_datetime_ordinal,
):
    transformer = DatetimeOrdinal(variables=["date_col_1"], drop_original=True)
    transformer.fit(df_datetime_ordinal)
    feature_names_out = transformer.get_feature_names_out(
        input_features=df_datetime_ordinal.columns.tolist()
    )

    expected_feature_names = ["date_col_1_ordinal", "date_col_2", "non_date_col"]
    assert sorted(feature_names_out) == sorted(expected_feature_names)


def test_datetime_ordinal_non_datetime_variable_in_transform(df_datetime_ordinal):
    transformer = DatetimeOrdinal(variables=["date_col_1"])
    transformer.fit(df_datetime_ordinal)
    # Create a new dataframe where 'date_col_1' is no longer datetime
    X_test = df_datetime_ordinal.copy()
    X_test["date_col_1"] = ["a", "b", "c", "d", "e"]

    with pytest.raises(ValueError):
        transformer.transform(X_test)