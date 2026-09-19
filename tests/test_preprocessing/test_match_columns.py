import datetime
import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.preprocessing import MatchVariables
from tests.backend_helpers import frame_to_dict, null_count

DOB = [datetime.datetime(2020, 2, 24, 0, minute) for minute in range(4)]

DATA_TRAIN = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
    "dob": DOB,
}

# lacks City and Age, has two extra variables and a different column order
DATA_TEST = {
    "extra_1": ["a", "b", "c", "d"],
    "Marks": [0.5, 0.4, 0.3, 0.2],
    "Name": ["sam", "fred", "peter", "bob"],
    "dob": DOB,
    "extra_2": [1, 2, 3, 4],
}

DATA_TRAIN_NA = {
    "Name": ["tom", None, "krish", "jack"],
    "City": ["London", "Manchester", None, "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, None, 0.7, 0.6],
}

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)


# init parameters
@pytest.mark.parametrize("fill_value", [[0, 1], None, {"a": 1}, (1,)])
def test_error_if_fill_value_not_allowed(fill_value):
    msg = f"fill_value takes integers, floats or strings. Got {fill_value} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        MatchVariables(fill_value=fill_value)


@pytest.mark.parametrize("missing_values", ["hola", 1, None, ["raise"]])
def test_error_if_missing_values_not_allowed(missing_values):
    msg = (
        "missing_values takes only values 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MatchVariables(missing_values=missing_values)


@pytest.mark.parametrize("match_dtypes", ["hallo", 1, None, [True]])
def test_error_if_match_dtypes_not_bool(match_dtypes):
    msg = (
        "match_dtypes takes only booleans True and False. "
        f"Got {match_dtypes} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MatchVariables(match_dtypes=match_dtypes)


@pytest.mark.parametrize("verbose", ["hallo", 1, None, [True]])
def test_error_if_verbose_not_bool(verbose):
    msg = f"verbose takes only booleans True and False. Got {verbose} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        MatchVariables(verbose=verbose)


@pytest.mark.parametrize(
    "fill_value, missing_values, match_dtypes, verbose",
    [
        (np.nan, "raise", False, True),
        (1, "ignore", True, False),
        (0.1, "raise", True, True),
        ("none", "ignore", False, False),
    ],
)
def test_init_param_assignment(fill_value, missing_values, match_dtypes, verbose):
    transformer = MatchVariables(
        fill_value=fill_value,
        missing_values=missing_values,
        match_dtypes=match_dtypes,
        verbose=verbose,
    )
    assert transformer.fill_value is fill_value
    assert transformer.missing_values == missing_values
    assert transformer.match_dtypes is match_dtypes
    assert transformer.verbose is verbose


# fit and transform
def test_fit_attributes(make_df):
    transformer = MatchVariables().fit(make_df(DATA_TRAIN))
    assert transformer.feature_names_in_ == ["Name", "City", "Age", "Marks", "dob"]
    assert transformer.n_features_in_ == 5
    assert not hasattr(transformer, "_dtype_dict")


@pytest.mark.parametrize(
    "fill_value, expected",
    [(np.nan, None), (1, 1), (0.1, 0.1), ("none", "none")],
)
def test_add_drop_and_reorder_variables(make_df, fill_value, expected):
    transformer = MatchVariables(fill_value=fill_value, verbose=False)
    transformer.fit(make_df(DATA_TRAIN))
    Xt = transformer.transform(make_df(DATA_TEST))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "Name": ["sam", "fred", "peter", "bob"],
        "City": [expected] * 4,
        "Age": [expected] * 4,
        "Marks": [0.5, 0.4, 0.3, 0.2],
        "dob": DOB,
    }
    assert transformer.get_feature_names_out() == [
        "Name",
        "City",
        "Age",
        "Marks",
        "dob",
    ]


@pytest.mark.parametrize(
    "fill_value, expected_dtype",
    [(np.nan, nw.Float64), (1, nw.Int64), (0.1, nw.Float64), ("none", nw.String)],
)
def test_dtype_of_added_variables(make_df, fill_value, expected_dtype):
    transformer = MatchVariables(fill_value=fill_value, verbose=False)
    transformer.fit(make_df(DATA_TRAIN))
    Xt = transformer.transform(make_df(DATA_TEST))

    schema = nw.from_native(Xt).schema
    assert schema["City"] == expected_dtype
    assert schema["Age"] == expected_dtype


def test_nan_fill_value_adds_missing_data(make_df):
    # polars treats NaN as a value, so the added variables must hold nulls.
    transformer = MatchVariables(verbose=False).fit(make_df(DATA_TRAIN))
    Xt = transformer.transform(make_df(DATA_TEST))
    assert null_count(Xt, "City") == 4
    assert null_count(Xt, "Age") == 4


def test_only_reorder_variables(make_df):
    X = make_df({"Age": [1, 2], "Name": ["a", "b"], "Marks": [0.1, 0.2]})
    train = make_df({"Name": ["c"], "Marks": [0.3], "Age": [3]})
    transformer = MatchVariables().fit(train)
    Xt = transformer.transform(X)

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["Name", "Marks", "Age"]
    assert frame_to_dict(Xt) == {
        "Name": ["a", "b"],
        "Marks": [0.1, 0.2],
        "Age": [1, 2],
    }


def test_no_variable_in_common(make_df):
    transformer = MatchVariables(verbose=False).fit(make_df({"a": [1], "b": ["x"]}))
    Xt = transformer.transform(make_df({"c": [1, 2]}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"a": [None, None], "b": [None, None]}


def test_transform_does_not_modify_input(make_df):
    X = make_df(DATA_TEST)
    transformer = MatchVariables(fill_value=0, verbose=False)
    transformer.fit(make_df(DATA_TRAIN))
    Xt = transformer.transform(X)

    assert list(X.columns) == ["extra_1", "Marks", "Name", "dob", "extra_2"]
    assert frame_to_dict(X) == DATA_TEST
    assert frame_to_dict(Xt)["City"] == [0, 0, 0, 0]


def test_verbose_print_out(capsys, make_df):
    transformer = MatchVariables(verbose=True).fit(make_df(DATA_TRAIN))
    transformer.transform(make_df(DATA_TEST))

    out, _ = capsys.readouterr()
    assert out == (
        "The following variables are added to the DataFrame: ['City', 'Age']\n"
        "The following variables are dropped from the DataFrame: "
        "['extra_1', 'extra_2']\n"
    )


def test_no_print_out_when_verbose_is_false(capsys, make_df):
    transformer = MatchVariables(fill_value=1, verbose=False, match_dtypes=True)
    transformer.fit(make_df(DATA_TRAIN))
    transformer.transform(make_df(DATA_TEST))

    out, _ = capsys.readouterr()
    assert out == ""


def test_raises_error_if_na_in_fit(make_df):
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        MatchVariables().fit(make_df(DATA_TRAIN_NA))


def test_raises_error_if_na_in_transform(make_df):
    transformer = MatchVariables().fit(make_df(DATA_TRAIN))
    X = make_df({**DATA_TRAIN, "Marks": [0.9, None, 0.7, 0.6]})
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(X)


def test_na_check_skips_variables_not_seen_in_fit(make_df):
    # the variables dropped in transform may contain NaN.
    transformer = MatchVariables(verbose=False).fit(make_df(DATA_TRAIN))
    Xt = transformer.transform(make_df({**DATA_TRAIN, "extra": [None, 1, 2, 3]}))
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == DATA_TRAIN


def test_na_check_skips_variables_missing_from_transform(make_df):
    # Bug reported in https://github.com/feature-engine/feature_engine/issues/789
    train = {
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
    }
    test = {
        "Name": ["tom", "sam", "nick"],
        "Age": [20, 22, 23],
        "Marks": [0.9, 0.7, 0.6],
        "Hobbies": ["tennis", "rugby", "football"],
    }
    transformer = MatchVariables().fit(make_df(train))
    Xt = transformer.transform(make_df(test))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "Name": ["tom", "sam", "nick"],
        "City": [None, None, None],
        "Age": [20, 22, 23],
        "Marks": [0.9, 0.7, 0.6],
    }


def test_missing_values_ignore(make_df):
    transformer = MatchVariables(missing_values="ignore", verbose=False)
    transformer.fit(make_df(DATA_TRAIN_NA))
    Xt = transformer.transform(make_df(DATA_TRAIN_NA))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == DATA_TRAIN_NA


def test_non_fitted_error(make_df):
    msg = (
        "This MatchVariables instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        MatchVariables().transform(make_df(DATA_TRAIN))


def test_match_dtypes_string_to_numbers(make_df):
    train = {"Age": [20, 21, 19, 18], "Marks": [0.9, 0.8, 0.7, 0.6]}
    test = {"Age": ["20", "21", "19", "18"], "Marks": ["0.9", "0.8", "0.7", "0.6"]}
    transformer = MatchVariables(match_dtypes=True).fit(make_df(train))
    Xt = transformer.transform(make_df(test))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == train
    schema = nw.from_native(Xt).schema
    assert schema["Age"] == nw.Int64
    assert schema["Marks"] == nw.Float64


def test_match_dtypes_numbers_to_string(make_df):
    train = {"Age": ["20", "21", "19", "18"], "Marks": ["0.9", "0.8", "0.7", "0.6"]}
    test = {"Age": [20, 21, 19, 18], "Marks": [0.9, 0.8, 0.7, 0.6]}
    transformer = MatchVariables(match_dtypes=True).fit(make_df(train))
    Xt = transformer.transform(make_df(test))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == train
    schema = nw.from_native(Xt).schema
    assert schema["Age"] == nw.String
    assert schema["Marks"] == nw.String


def test_match_dtypes_string_to_datetime(make_df):
    test = {"dob": [str(date) for date in DOB]}
    transformer = MatchVariables(match_dtypes=True).fit(make_df({"dob": DOB}))
    Xt = transformer.transform(make_df(test))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"dob": DOB}
    assert nw.from_native(Xt).schema["dob"] == nw.Datetime


def test_match_dtypes_datetime_to_string(make_df):
    train = {"dob": [str(date) for date in DOB]}
    transformer = MatchVariables(match_dtypes=True).fit(make_df(train))
    Xt = transformer.transform(make_df({"dob": DOB}))

    assert isinstance(Xt, make_df)
    assert nw.from_native(Xt).schema["dob"] == nw.String
    # polars adds the microseconds to the string, pandas doesn't.
    for value, date in zip(frame_to_dict(Xt)["dob"], DOB):
        assert value.startswith(str(date))


def test_match_dtypes_integers_and_floats(make_df):
    train = {"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}
    test = {"a": [1.9, 2.0, 3.5], "b": [4, 5, 6]}
    transformer = MatchVariables(match_dtypes=True).fit(make_df(train))
    Xt = transformer.transform(make_df(test))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    schema = nw.from_native(Xt).schema
    assert schema["a"] == nw.Int64
    assert schema["b"] == nw.Float64


def test_match_dtypes_of_added_variables(make_df):
    transformer = MatchVariables(fill_value=1, match_dtypes=True, verbose=False)
    transformer.fit(make_df(DATA_TRAIN))
    Xt = transformer.transform(make_df(DATA_TEST))

    assert frame_to_dict(Xt)["City"] == ["1", "1", "1", "1"]
    assert nw.from_native(Xt).schema["City"] == nw.String


def test_match_dtypes_of_added_integer_and_boolean_variables(make_df):
    # the added variables are missing, so they need types that allow missing data
    train = {"int": [1, 2, 3], "bool": [True, False, True], "float": [0.1, 0.2, 0.3]}
    transformer = MatchVariables(match_dtypes=True, verbose=False)
    transformer.fit(make_df(train))
    Xt = transformer.transform(make_df({"float": [0.4, 0.5]}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "int": [None, None],
        "bool": [None, None],
        "float": [0.4, 0.5],
    }
    schema = nw.from_native(Xt).schema
    assert schema["int"] == nw.Int64
    assert schema["bool"] == nw.Boolean


@pytest.mark.parametrize("test_is_categorical", [False, True])
def test_match_dtypes_categories(make_df, test_is_categorical):
    # values not in the categories seen in fit become missing data.
    train = (
        nw.from_native(make_df({"Name": ["nick", "krish", "jack"]}))
        .with_columns(nw.col("Name").cast(nw.Enum(["jack", "krish", "nick"])))
        .to_native()
    )
    test = nw.from_native(make_df({"Name": ["tom", "nick", "jack", None]}))
    if test_is_categorical is True:
        test = test.with_columns(nw.col("Name").cast(nw.Enum(["jack", "nick", "tom"])))

    transformer = MatchVariables(
        missing_values="ignore", match_dtypes=True, verbose=False
    ).fit(train)
    Xt = transformer.transform(test.to_native())

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"Name": [None, "nick", "jack", None]}
    assert nw.from_native(Xt).schema["Name"] == nw.from_native(train).schema["Name"]


def test_match_dtypes_verbose_print_out(capsys, make_df):
    transformer = MatchVariables(match_dtypes=True, verbose=True)
    transformer.fit(make_df({"Age": [20, 21]}))
    capsys.readouterr()
    transformer.transform(make_df({"Age": [1.0, 2.0]}))

    out, _ = capsys.readouterr()
    if make_df is pd.DataFrame:
        assert out == "The Age dtype is changing from  float64 to int64\n"
    else:
        assert out == "The Age dtype is changing from  Float64 to Int64\n"


def test_pandas_dtype_dict_and_categories():
    train = pd.DataFrame(
        {"Name": DATA_TRAIN["Name"], "City": DATA_TRAIN["City"]}
    ).astype("category")
    # the last row holds jack and Bristol, so test lacks these categories.
    test = train.iloc[:-1].astype(str).astype("category")

    transformer = MatchVariables(match_dtypes=True, verbose=False).fit(train)
    Xt = transformer.transform(test)

    assert transformer._dtype_dict == {
        "Name": pd.CategoricalDtype(
            categories=["jack", "krish", "nick", "tom"], ordered=False
        ),
        "City": pd.CategoricalDtype(
            categories=["Bristol", "Liverpool", "London", "Manchester"], ordered=False
        ),
    }
    pd.testing.assert_series_equal(Xt.dtypes, train.dtypes)
    pd.testing.assert_frame_equal(Xt, train.iloc[:-1])


def test_polars_dtype_dict():
    train = pl.DataFrame({"a": [1], "b": ["x"], "c": [DOB[0]]})
    transformer = MatchVariables(match_dtypes=True).fit(train)
    assert transformer._dtype_dict == {
        "a": nw.Int64(),
        "b": nw.String(),
        "c": nw.Datetime("us"),
    }


def test_polars_match_dtypes_string_to_date():
    # pandas has no date dtype, it stores dates as objects.
    dates = [datetime.date(2020, 2, day) for day in range(24, 27)]
    transformer = MatchVariables(match_dtypes=True).fit(pl.DataFrame({"d": dates}))
    Xt = transformer.transform(
        pl.DataFrame({"d": ["2020-02-24", "2020-02-25", "2020-02-26"]})
    )
    assert Xt.schema["d"] == pl.Date
    assert Xt["d"].to_list() == dates


def test_pandas_integer_column_names():
    train = pd.DataFrame({0: [1, 2, 3], 1: ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [4.0, 5.0, 6.0], 5: [0, 0, 0], 1: ["d", "e", "f"]})
    transformer = MatchVariables(fill_value=0, match_dtypes=True, verbose=False)
    Xt = transformer.fit(train).transform(test)

    expected = pd.DataFrame({0: [0, 0, 0], 1: ["d", "e", "f"], "x": [4.0, 5.0, 6.0]})
    pd.testing.assert_frame_equal(Xt, expected)
    assert transformer.feature_names_in_ == [0, 1, "x"]


def test_pandas_index_is_kept():
    transformer = MatchVariables(verbose=False).fit(pd.DataFrame(DATA_TRAIN))
    Xt = transformer.transform(pd.DataFrame(DATA_TEST, index=[10, 20, 30, 40]))

    expected = pd.DataFrame(
        {
            "Name": ["sam", "fred", "peter", "bob"],
            "City": [np.nan] * 4,
            "Age": [np.nan] * 4,
            "Marks": [0.5, 0.4, 0.3, 0.2],
            "dob": DOB,
        },
        index=[10, 20, 30, 40],
    )
    pd.testing.assert_frame_equal(Xt, expected)
