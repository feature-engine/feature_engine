import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.imputation import EndTailImputer

# Missing values are written as `None`, not `np.nan`: polars treats np.nan as
# a real float value (not a null), so mean/std/quantile would NOT skip it,
# unlike pandas' NaN-as-missing default. `None` becomes a null on both
# backends and is skipped by both, keeping the two code paths comparable.
DATA = {
    "Name": ["tom", "nick", "krish", None, "peter", None, "fred", "sam"],
    "City": [
        "London",
        "Manchester",
        None,
        None,
        "London",
        "London",
        "Bristol",
        "Manchester",
    ],
    "Studies": [
        "Bachelor",
        "Bachelor",
        None,
        None,
        "Bachelor",
        "PhD",
        "None",
        "Masters",
    ],
    "Age": [20, 21, 19, None, 23, 40, 41, 37],
    "Marks": [0.9, 0.8, 0.7, None, 0.3, None, 0.8, 0.6],
}


def _none_to_nan(values):
    # Missing values print as None for polars, NaN for pandas float columns
    # - both mean "missing" here, so normalize both sides before comparing.
    return [np.nan if v is None else v for v in values]


def assert_df_equal(X, expected: dict, abs_tol: float = 1e-5) -> None:
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        assert _none_to_nan(result[col]) == pytest.approx(
            _none_to_nan(values), abs=abs_tol, nan_ok=True
        )


def _missing_count(X, columns) -> int:
    nw_X = nw.from_native(X, eager_only=True)
    return sum(int(nw_X.get_column(c).is_null().sum()) for c in columns)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_find_variables_and_gaussian_imputation_on_right_tail(make_df):
    df = make_df(DATA)
    imputer = EndTailImputer(
        imputation_method="gaussian", tail="right", fold=3, variables=None
    )
    X_transformed = imputer.fit_transform(df)

    # test init params
    assert imputer.imputation_method == "gaussian"
    assert imputer.tail == "right"
    assert imputer.fold == 3
    assert imputer.variables is None
    # test fit attr
    assert imputer.variables_ == ["Age", "Marks"]
    assert imputer.n_features_in_ == 5
    rounded = {k: round(v, 3) for k, v in imputer.imputer_dict_.items()}
    assert rounded == {"Age": 58.949, "Marks": 1.324}

    # transform output: indicated vars ==> no NA, not indicated vars with NA
    assert _missing_count(X_transformed, ["Age", "Marks"]) == 0
    assert _missing_count(X_transformed, ["City", "Name"]) > 0

    expected = dict(DATA)
    expected["Age"] = [20, 21, 19, 58.94908118478389, 23, 40, 41, 37]
    expected["Marks"] = [
        0.9, 0.8, 0.7, 1.3244261503263175, 0.3, 1.3244261503263175, 0.8, 0.6,
    ]
    assert_df_equal(X_transformed, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_enters_variables_and_iqr_imputation_on_right_tail(make_df):
    df = make_df(DATA)
    imputer = EndTailImputer(
        imputation_method="iqr", tail="right", fold=1.5, variables=["Age", "Marks"]
    )
    X_transformed = imputer.fit_transform(df)

    assert imputer.imputer_dict_ == {"Age": 65.5, "Marks": 1.0625}
    assert _missing_count(X_transformed, ["Age", "Marks"]) == 0

    expected = dict(DATA)
    expected["Age"] = [20, 21, 19, 65.5, 23, 40, 41, 37]
    expected["Marks"] = [0.9, 0.8, 0.7, 1.0625, 0.3, 1.0625, 0.8, 0.6]
    assert_df_equal(X_transformed, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_enters_variables_and_max_value_imputation(make_df):
    df = make_df(DATA)
    imputer = EndTailImputer(
        imputation_method="max", tail="right", fold=2, variables=["Age", "Marks"]
    )
    imputer.fit(df)
    assert imputer.imputer_dict_ == {"Age": 82.0, "Marks": 1.8}


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_select_variables_and_gaussian_imputation_on_left_tail(make_df):
    df = make_df(DATA)
    imputer = EndTailImputer(imputation_method="gaussian", tail="left", fold=3)
    imputer.fit(df)
    rounded = {k: round(v, 3) for k, v in imputer.imputer_dict_.items()}
    assert rounded == {"Age": -1.521, "Marks": 0.042}


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_enters_variables_and_iqr_imputation_on_left_tail(make_df):
    df = make_df(DATA)
    imputer = EndTailImputer(
        imputation_method="iqr", tail="left", fold=1.5, variables=["Age", "Marks"]
    )
    imputer.fit(df)
    assert imputer.imputer_dict_["Age"] == -6.5
    assert np.round(imputer.imputer_dict_["Marks"], 3) == np.round(
        0.36249999999999993, 3
    )


def test_error_when_imputation_method_is_not_permitted():
    with pytest.raises(ValueError, match="imputation_method takes only values"):
        EndTailImputer(imputation_method="arbitrary")


def test_error_when_tail_is_string():
    with pytest.raises(ValueError, match="tail takes only values"):
        EndTailImputer(tail="arbitrary")


def test_error_when_fold_is_1():
    with pytest.raises(ValueError, match="fold takes only positive numbers"):
        EndTailImputer(fold=-1)
