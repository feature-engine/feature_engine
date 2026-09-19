import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from feature_engine.creation.base_creation import BaseCreation

BASIC_DATA = {
    "var_a": [1, 2, 3, 4],
    "var_b": [10, 20, 30, 40],
    "var_c": [100, 200, 300, 400],
}


class StubCreation(BaseCreation):
    def __init__(self, variables=None, missing_values="raise", drop_original=False):
        self.variables = variables
        super().__init__(missing_values=missing_values, drop_original=drop_original)

    def transform(self, X):
        return self._check_transform_input_and_state(X)


class StubWithReference(StubCreation):
    def __init__(
        self, reference, variables=None, missing_values="raise", drop_original=False
    ):
        self.reference = reference
        super().__init__(
            variables=variables,
            missing_values=missing_values,
            drop_original=drop_original,
        )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_transform_round_trip(make_df):
    X = make_df(BASIC_DATA)
    transformer = StubCreation()
    transformer.fit(X)

    assert transformer.variables_ == ["var_a", "var_b", "var_c"]
    assert transformer.feature_names_in_ == ["var_a", "var_b", "var_c"]
    assert transformer.n_features_in_ == 3

    Xt = transformer.transform(X)
    assert list(nw.from_native(Xt, eager_only=True).columns) == [
        "var_a",
        "var_b",
        "var_c",
    ]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_reorders_columns_to_match_fit(make_df):
    X = make_df(BASIC_DATA)
    transformer = StubCreation()
    transformer.fit(X)

    reordered = make_df(
        {
            "var_c": BASIC_DATA["var_c"],
            "var_a": BASIC_DATA["var_a"],
            "var_b": BASIC_DATA["var_b"],
        }
    )
    Xt = transformer.transform(reordered)
    assert list(nw.from_native(Xt, eager_only=True).columns) == [
        "var_a",
        "var_b",
        "var_c",
    ]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_when_column_count_differs(make_df):
    X = make_df(BASIC_DATA)
    transformer = StubCreation()
    transformer.fit(X)

    X_fewer_cols = make_df(
        {"var_a": BASIC_DATA["var_a"], "var_b": BASIC_DATA["var_b"]}
    )
    msg = (
        "The number of columns in this dataset is different from the one used to "
        "fit this transformer"
    )
    with pytest.raises(ValueError, match=msg):
        transformer.transform(X_fewer_cols)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_missing_values_raise_vs_ignore(make_df):
    data_with_na = {**BASIC_DATA, "var_a": [1, None, 3, 4]}
    X = make_df(data_with_na)

    transformer_raise = StubCreation(missing_values="raise")
    msg = "Some of the variables in the dataset contain NaN"
    with pytest.raises(ValueError, match=msg):
        transformer_raise.fit(X)

    transformer_ignore = StubCreation(missing_values="ignore")
    transformer_ignore.fit(X)
    Xt = transformer_ignore.transform(X)
    assert len(nw.from_native(Xt, eager_only=True)) == 4


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_reference_attribute_is_checked_in_fit(make_df):
    X = make_df(BASIC_DATA)
    transformer = StubWithReference(reference=["var_a"])
    transformer.fit(X)
    assert transformer.variables_ == ["var_a", "var_b", "var_c"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_reference_must_be_numerical(make_df):
    X = make_df({**BASIC_DATA, "var_d": ["a", "b", "c", "d"]})
    transformer = StubWithReference(reference=["var_d"])
    msg = "Some of the variables are not numerical"
    with pytest.raises(TypeError, match=msg):
        transformer.fit(X)
