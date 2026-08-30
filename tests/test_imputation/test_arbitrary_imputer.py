import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from feature_engine.imputation import ArbitraryImputer, ArbitraryNumberImputer

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
    "Age": [20.0, 21.0, 19.0, None, 23.0, 40.0, 41.0, 37.0],
    "Marks": [0.9, 0.8, 0.7, None, 0.3, None, 0.8, 0.6],
}


def _null_count(X, col) -> int:
    return nw.from_native(X, eager_only=True)[col].is_null().sum()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_impute_with_99_and_automatically_select_variables(make_df):
    X = make_df(DATA)
    imputer = ArbitraryImputer(arbitrary_number=99, variables=None)
    X_transformed = imputer.fit_transform(X)

    # test init params
    assert imputer.arbitrary_number == 99
    assert imputer.variables is None

    # test fit attributes
    assert imputer.variables_ == ["Age", "Marks"]
    assert imputer.n_features_in_ == 4
    assert imputer.imputer_dict_ == {"Age": 99, "Marks": 99}

    # selected variables should not contain NA, non-selected should still
    assert _null_count(X_transformed, "Age") == 0
    assert _null_count(X_transformed, "Marks") == 0
    assert _null_count(X_transformed, "Name") > 0
    assert _null_count(X_transformed, "City") > 0

    result = nw.from_native(X_transformed, eager_only=True).to_dict(as_series=False)
    assert result["Age"] == [20.0, 21.0, 19.0, 99.0, 23.0, 40.0, 41.0, 37.0]
    assert result["Marks"] == [0.9, 0.8, 0.7, 99.0, 0.3, 99.0, 0.8, 0.6]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_impute_with_1_and_single_variable_entered_by_user(make_df):
    X = make_df(DATA)
    imputer = ArbitraryImputer(arbitrary_number=-1, variables=["Age"])
    X_transformed = imputer.fit_transform(X)

    # test init params
    assert imputer.arbitrary_number == -1
    assert imputer.variables == ["Age"]

    # test fit attributes
    assert imputer.variables_ == ["Age"]
    assert imputer.n_features_in_ == 4
    assert imputer.imputer_dict_ == {"Age": -1}

    assert _null_count(X_transformed, "Age") == 0
    result = nw.from_native(X_transformed, eager_only=True).to_dict(as_series=False)
    assert result["Age"] == [20.0, 21.0, 19.0, -1.0, 23.0, 40.0, 41.0, 37.0]


def test_error_when_arbitrary_number_is_string():
    with pytest.raises(ValueError):
        ArbitraryImputer(arbitrary_number="arbitrary")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_dictionary_of_imputation_values(make_df):
    X = make_df(DATA)
    imputer = ArbitraryImputer(imputer_dict={"Age": -42, "Marks": -999})
    X_transformed = imputer.fit_transform(X)

    # test fit params
    assert imputer.n_features_in_ == 4
    assert imputer.imputer_dict_ == {"Age": -42, "Marks": -999}

    assert _null_count(X_transformed, "Age") == 0
    assert _null_count(X_transformed, "Marks") == 0
    assert _null_count(X_transformed, "Name") > 0
    assert _null_count(X_transformed, "City") > 0

    result = nw.from_native(X_transformed, eager_only=True).to_dict(as_series=False)
    assert result["Age"] == [20.0, 21.0, 19.0, -42.0, 23.0, 40.0, 41.0, 37.0]
    assert result["Marks"] == [0.9, 0.8, 0.7, -999.0, 0.3, -999.0, 0.8, 0.6]


def test_imputer_error_when_dictionary_value_is_string():
    with pytest.raises(ValueError):
        ArbitraryImputer(imputer_dict={"Age": "arbitrary_number"})


def test_arbitrary_number_imputer_is_deprecated():
    """ArbitraryNumberImputer should emit a FutureWarning and still work."""
    with pytest.warns(FutureWarning, match="ArbitraryNumberImputer was deprecated"):
        imputer = ArbitraryNumberImputer(arbitrary_number=99)
    assert isinstance(imputer, ArbitraryImputer)
    assert imputer.arbitrary_number == 99
