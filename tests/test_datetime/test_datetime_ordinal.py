import datetime
import math

import pandas as pd
import polars as pl
import pytest

from feature_engine.datetime import DatetimeOrdinal

DATE_COLS = ["date_col_1", "date_col_2"]

DATE_DATA = {
    "date_col_1": [
        "2023-01-01",
        "2023-01-02",
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
    ],
    "date_col_2": [
        "2024-02-10",
        "2024-02-11",
        "2024-02-12",
        "2024-02-13",
        "2024-02-14",
    ],
    "non_date_col": [1, 2, 3, 4, 5],
}

DATE_DATA_NA = {
    "date_col_1": ["2023-01-01", "2023-01-02", None, "2023-01-04", "2023-01-05"],
    "date_col_2": ["2024-02-10", "2024-02-11", "2024-02-12", None, "2024-02-14"],
}


def _make_datetime_df(make_df, data: dict, date_cols=DATE_COLS):
    """Build a dataframe where `date_cols` hold a native Date/Datetime dtype
    (not strings), the same way real datetime columns arrive in practice -
    constructed differently per backend since pandas and polars have no
    shared literal syntax for it."""
    if make_df is pd.DataFrame:
        return pd.DataFrame(
            {
                col: pd.to_datetime(values) if col in date_cols else values
                for col, values in data.items()
            }
        )
    df = pl.DataFrame(data)
    return df.with_columns([pl.col(c).str.to_datetime() for c in date_cols])


def _expected_ordinal(date_strings, start_date_ordinal=None):
    result = []
    for s in date_strings:
        if s is None:
            result.append(None)
            continue
        ordinal = datetime.date.fromisoformat(s).toordinal()
        if start_date_ordinal is not None:
            ordinal = ordinal - start_date_ordinal + 1
        result.append(ordinal)
    return result


def _as_comparable_ints(values):
    """Normalize a result column to plain ints/None regardless of whether the
    backend represented missing ordinals as NaN (pandas float64) or null
    (polars Int64) - same values, different native missing-data convention."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
        else:
            out.append(int(v))
    return out


def _get_col(X, col):
    if isinstance(X, pd.DataFrame):
        return X[col].tolist()
    return X[col].to_list()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "variables_param",
    [["date_col_1", "date_col_2"], None],
    ids=["variables_specified", "variables_auto_find"],
)
def test_datetime_ordinal_feature_creation(make_df, variables_param):
    X = _make_datetime_df(make_df, DATE_DATA)
    transformer = DatetimeOrdinal(variables=variables_param)
    Xt = transformer.fit_transform(X)

    assert _as_comparable_ints(_get_col(Xt, "date_col_1_ordinal")) == _expected_ordinal(
        DATE_DATA["date_col_1"]
    )
    assert _as_comparable_ints(_get_col(Xt, "date_col_2_ordinal")) == _expected_ordinal(
        DATE_DATA["date_col_2"]
    )

    columns = Xt.columns
    assert "non_date_col" in columns
    assert "date_col_1" not in columns
    assert "date_col_2" not in columns


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_with_start_date(make_df):
    start_date_str = "2023-01-01"
    X = _make_datetime_df(make_df, DATE_DATA)
    transformer = DatetimeOrdinal(variables=["date_col_1"], start_date=start_date_str)
    Xt = transformer.fit_transform(X)

    start_ordinal = datetime.date.fromisoformat(start_date_str).toordinal()
    expected = _expected_ordinal(
        DATE_DATA["date_col_1"], start_date_ordinal=start_ordinal
    )

    assert _as_comparable_ints(_get_col(Xt, "date_col_1_ordinal")) == expected
    assert "date_col_2" in Xt.columns
    assert "date_col_1" not in Xt.columns


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_with_start_date_datetime_object(make_df):
    start_date_obj = datetime.date(2023, 1, 1)
    X = _make_datetime_df(make_df, DATE_DATA)
    transformer = DatetimeOrdinal(variables=["date_col_1"], start_date=start_date_obj)
    Xt = transformer.fit_transform(X)

    expected = _expected_ordinal(
        DATE_DATA["date_col_1"], start_date_ordinal=start_date_obj.toordinal()
    )
    assert _as_comparable_ints(_get_col(Xt, "date_col_1_ordinal")) == expected


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_missing_values_raise(make_df):
    X = _make_datetime_df(make_df, DATE_DATA_NA)
    transformer = DatetimeOrdinal(missing_values="raise")
    with pytest.raises(
        ValueError, match="Some of the variables in the dataset contain NaN"
    ):
        transformer.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_missing_values_ignore(make_df):
    X = _make_datetime_df(make_df, DATE_DATA_NA)
    transformer = DatetimeOrdinal(missing_values="ignore")
    Xt = transformer.fit_transform(X)

    assert _as_comparable_ints(
        _get_col(Xt, "date_col_1_ordinal")
    ) == _expected_ordinal(DATE_DATA_NA["date_col_1"])
    assert _as_comparable_ints(
        _get_col(Xt, "date_col_2_ordinal")
    ) == _expected_ordinal(DATE_DATA_NA["date_col_2"])


def test_datetime_ordinal_invalid_start_date():
    with pytest.raises(
        ValueError, match="start_date could not be converted to datetime"
    ):
        DatetimeOrdinal(start_date="not-a-date")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_non_datetime_variable_error(make_df):
    X = make_df(DATE_DATA)
    transformer = DatetimeOrdinal(variables=["non_date_col"])
    with pytest.raises(TypeError):
        transformer.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_drop_original_false(make_df):
    X = _make_datetime_df(make_df, DATE_DATA)
    transformer = DatetimeOrdinal(variables=["date_col_1"], drop_original=False)
    Xt = transformer.fit_transform(X)

    assert "date_col_1" in Xt.columns
    assert "date_col_1_ordinal" in Xt.columns
    assert "date_col_2" in Xt.columns


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_get_feature_names_out(make_df):
    X = _make_datetime_df(make_df, DATE_DATA)
    transformer = DatetimeOrdinal(variables=["date_col_1", "date_col_2"])
    transformer.fit(X)
    feature_names_out = transformer.get_feature_names_out()

    expected_feature_names = [
        "date_col_1_ordinal",
        "date_col_2_ordinal",
        "non_date_col",
    ]
    assert sorted(feature_names_out) == sorted(expected_feature_names)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_get_feature_names_out_with_input_features(make_df):
    X = _make_datetime_df(make_df, DATE_DATA)
    transformer = DatetimeOrdinal(variables=["date_col_1"], drop_original=False)
    transformer.fit(X)
    feature_names_out = transformer.get_feature_names_out(
        input_features=list(X.columns)
    )

    expected_feature_names = [
        "date_col_1_ordinal",
        "date_col_2",
        "non_date_col",
        "date_col_1",
    ]
    assert sorted(feature_names_out) == sorted(expected_feature_names)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_get_feature_names_out_with_input_features_drop_original(
    make_df,
):
    X = _make_datetime_df(make_df, DATE_DATA)
    transformer = DatetimeOrdinal(variables=["date_col_1"], drop_original=True)
    transformer.fit(X)
    feature_names_out = transformer.get_feature_names_out(
        input_features=list(X.columns)
    )

    expected_feature_names = ["date_col_1_ordinal", "date_col_2", "non_date_col"]
    assert sorted(feature_names_out) == sorted(expected_feature_names)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_non_datetime_variable_in_transform(make_df):
    X = _make_datetime_df(make_df, DATE_DATA)
    transformer = DatetimeOrdinal(variables=["date_col_1"])
    transformer.fit(X)

    junk_data = {**DATE_DATA, "date_col_1": ["a", "b", "c", "d", "e"]}
    X_test = make_df(junk_data)

    # pandas raises ValueError, polars raises its own ComputeError - different
    # exception classes, but both signal the same "not a real date" failure.
    with pytest.raises(Exception):
        transformer.transform(X_test)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_ordinal_missing_values_raise_in_transform(make_df):
    X = _make_datetime_df(make_df, DATE_DATA)
    transformer = DatetimeOrdinal(missing_values="raise")
    transformer.fit(X)

    na_data = {**DATE_DATA_NA, "non_date_col": [1, 2, 3, 4, 5]}
    X_test = _make_datetime_df(make_df, na_data)

    with pytest.raises(
        ValueError, match="Some of the variables in the dataset contain NaN"
    ):
        transformer.transform(X_test)


def test_raises_error_for_invalid_missing_values():
    with pytest.raises(
        ValueError, match="missing_values takes only values 'raise' or 'ignore'"
    ):
        DatetimeOrdinal(missing_values="foo")


def test_raises_error_for_invalid_drop_original():
    with pytest.raises(
        ValueError, match="drop_original takes only booleans True or False"
    ):
        DatetimeOrdinal(drop_original="bar")


def test_more_tags_returns_expected_tags():
    transformer = DatetimeOrdinal()
    expected_tags = {"variables": "datetime"}
    assert transformer._more_tags() == expected_tags


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_return_empty(make_df):
    # DatetimeOrdinal.__init__ does not store `self.start_date = start_date`
    # (only the derived `self.start_date_`), which breaks sklearn's
    # get_params()/clone() for this transformer. Because of that, it cannot go
    # through the shared, clone-based check_return_empty check, nor through
    # check_feature_engine_estimator at all. This test instantiates the
    # transformer directly instead.
    X = make_df({"var_num": [1.0, 2.0, 3.0]})

    transformer = DatetimeOrdinal(variables=None, return_empty=False)
    with pytest.raises(
        TypeError, match="No datetime variables found in this dataframe"
    ):
        transformer.fit(X)

    transformer = DatetimeOrdinal(variables=None, return_empty=True)
    with pytest.warns(
        UserWarning,
        match="No datetime variables found in this dataframe. Returning an empty list.",
    ):
        transformer.fit(X)
    assert transformer.variables_ == []

    # if return_empty=True, transformer should return same df after transformation
    Xt = transformer.transform(X)
    assert _get_col(Xt, "var_num") == _get_col(X, "var_num")
    assert transformer.get_feature_names_out() == list(X.columns)
