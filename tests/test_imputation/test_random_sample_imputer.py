# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from feature_engine.imputation import RandomSampleImputer
from feature_engine.imputation.random_sample import _define_seed

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


def _null_count(X, col):
    return nw.from_native(X, eager_only=True)[col].null_count()


def _values(X, col):
    return nw.from_native(X, eager_only=True)[col].to_list()


def _pool(X, col):
    # values available for the imputer to sample from, in the copy of the
    # training data it stores at fit()
    return set(nw.from_native(X, eager_only=True)[col].drop_nulls().to_list())


def _is_missing(v):
    return v is None or (isinstance(v, float) and v != v)


def _same_values(a, b):
    # element-wise equality that treats None and float NaN as equal missing
    # markers, since pandas' NaN and polars'/narwhals' None represent the
    # same "missing" concept but compare unequal with plain `==`.
    return len(a) == len(b) and all(
        (_is_missing(x) and _is_missing(y)) or x == y for x, y in zip(a, b)
    )


def test_define_seed(df_vartypes):
    # _define_seed uses pandas' .loc label-based row access, so it is only
    # ever called from the pandas branch of transform() - it is inherently
    # pandas-only, unlike the rest of the transformer.
    assert _define_seed(df_vartypes, 0, ["Age", "Marks"], how="add") == 21
    assert _define_seed(df_vartypes, 0, ["Age", "Marks"], how="multiply") == 18
    assert _define_seed(df_vartypes, 2, ["Age", "Marks"], how="add") == 20
    assert _define_seed(df_vartypes, 2, ["Age", "Marks"], how="multiply") == 13
    assert _define_seed(df_vartypes, 1, ["Age"], how="add") == 21
    assert _define_seed(df_vartypes, 3, ["Marks"], how="multiply") == 1


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_general_seed_plus_automatically_select_variables(make_df):
    df_na = make_df(DATA)
    imputer = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed = imputer.fit_transform(df_na)

    # test init params
    assert imputer.variables is None
    assert imputer.random_state == 5
    assert imputer.seed == "general"

    # test fit attrs
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 5
    for col in imputer.variables_:
        assert _same_values(_values(imputer.X_, col), _values(df_na, col))

    # no missing data left in any imputed variable
    for col in imputer.variables_:
        assert _null_count(X_transformed, col) == 0
        # every value used to fill NA came from the training data itself
        assert set(_values(X_transformed, col)) <= _pool(df_na, col)

    # pandas' and narwhals/polars' sample() use different RNGs, so a fixed
    # seed does not draw the same values across backends - only same seed +
    # same backend is a reproducibility guarantee. Verify that guarantee.
    imputer2 = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed2 = imputer2.fit_transform(df_na)
    for col in imputer.variables_:
        assert _values(X_transformed, col) == _values(X_transformed2, col)


def test_pandas_general_seed_reproduces_historic_values(df_na):
    # Regression guard for the pandas fast-path specifically: transform()'s
    # pandas branch is untouched code (still pandas' own .sample()/.loc), so
    # for a fixed seed it must keep drawing the exact same values it drew
    # before this narwhals migration. These literal values are inherently
    # pandas-RNG-specific (see class docstring) and cannot be reproduced by
    # any other backend, so this check is legitimately pandas-only.
    imputer = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed = imputer.fit_transform(df_na)

    ref = {
        "Name": ["tom", "nick", "krish", "peter", "peter", "sam", "fred", "sam"],
        "City": [
            "London",
            "Manchester",
            "London",
            "Manchester",
            "London",
            "London",
            "Bristol",
            "Manchester",
        ],
        "Studies": [
            "Bachelor",
            "Bachelor",
            "PhD",
            "Masters",
            "Bachelor",
            "PhD",
            "None",
            "Masters",
        ],
        "Age": [20, 21, 19, 23, 23, 40, 41, 37],
        "Marks": [0.9, 0.8, 0.7, 0.3, 0.3, 0.6, 0.8, 0.6],
        "dob": pd.date_range("2020-02-24", periods=8, freq="min"),
    }
    ref = pd.DataFrame(ref)

    pd.testing.assert_frame_equal(X_transformed, ref, check_dtype=False)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_seed_per_observation_and_multiple_variables_in_random_state(make_df):
    # Note: the variables used as seed should not have missing data, this I fill
    data = dict(DATA)
    data["Marks"] = [v if v is not None else 1 for v in data["Marks"]]
    data["Age"] = [v if v is not None else 1 for v in data["Age"]]
    df_na = make_df(data)

    imputer = RandomSampleImputer(
        variables=["City", "Studies"], random_state=["Marks", "Age"], seed="observation"
    )
    X_transformed = imputer.fit_transform(df_na)

    assert imputer.variables == ["City", "Studies"]
    assert imputer.random_state == ["Marks", "Age"]
    assert imputer.seed == "observation"
    for col in ["City", "Studies"]:
        assert _same_values(_values(imputer.X_, col), _values(df_na, col))
        assert _null_count(X_transformed, col) == 0
        assert set(_values(X_transformed, col)) <= _pool(df_na, col)
    # variables not selected for imputation are untouched
    assert _same_values(_values(X_transformed, "Age"), _values(df_na, "Age"))

    # same seed, same backend -> same result
    imputer2 = RandomSampleImputer(
        variables=["City", "Studies"], random_state=["Marks", "Age"], seed="observation"
    )
    X_transformed2 = imputer2.fit_transform(df_na)
    for col in ["City", "Studies"]:
        assert _values(X_transformed, col) == _values(X_transformed2, col)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_seed_per_observation_plus_product_of_seeding_variables(make_df):
    data = dict(DATA)
    data["Marks"] = [v if v is not None else 1 for v in data["Marks"]]
    data["Age"] = [v if v is not None else 1 for v in data["Age"]]
    df_na = make_df(data)

    imputer = RandomSampleImputer(
        variables=["City", "Studies"],
        random_state=["Marks", "Age"],
        seed="observation",
        seeding_method="multiply",
    )
    X_transformed = imputer.fit_transform(df_na)

    assert imputer.variables == ["City", "Studies"]
    assert imputer.random_state == ["Marks", "Age"]
    assert imputer.seed == "observation"
    for col in ["City", "Studies"]:
        assert _same_values(_values(imputer.X_, col), _values(df_na, col))
        assert _null_count(X_transformed, col) == 0
        assert set(_values(X_transformed, col)) <= _pool(df_na, col)

    imputer2 = RandomSampleImputer(
        variables=["City", "Studies"],
        random_state=["Marks", "Age"],
        seed="observation",
        seeding_method="multiply",
    )
    X_transformed2 = imputer2.fit_transform(df_na)
    for col in ["City", "Studies"]:
        assert _values(X_transformed, col) == _values(X_transformed2, col)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_seed_per_observation_with_only_1_variable_as_seed(make_df):
    data = dict(DATA)
    data["Age"] = [v if v is not None else 1 for v in data["Age"]]
    df_na = make_df(data)

    imputer = RandomSampleImputer(
        variables=["City", "Studies"], random_state="Age", seed="observation"
    )
    X_transformed = imputer.fit_transform(df_na)

    assert imputer.random_state == ["Age"]
    for col in ["City", "Studies"]:
        assert _same_values(_values(imputer.X_, col), _values(df_na, col))
        assert _null_count(X_transformed, col) == 0
        assert set(_values(X_transformed, col)) <= _pool(df_na, col)

    imputer2 = RandomSampleImputer(
        variables=["City", "Studies"], random_state="Age", seed="observation"
    )
    X_transformed2 = imputer2.fit_transform(df_na)
    for col in ["City", "Studies"]:
        assert _values(X_transformed, col) == _values(X_transformed2, col)


def test_error_if_seed_not_permitted_value():
    with pytest.raises(ValueError):
        RandomSampleImputer(seed="arbitrary")


def test_error_if_seeding_method_not_permitted_value():
    with pytest.raises(ValueError):
        RandomSampleImputer(seeding_method="arbitrary")


def test_error_if_random_state_takes_not_permitted_value():
    with pytest.raises(ValueError):
        RandomSampleImputer(seed="general", random_state="arbitrary")


def test_error_if_random_state_is_none_when_seed_is_observation():
    with pytest.raises(ValueError):
        RandomSampleImputer(seed="observation", random_state=None)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_random_state_is_string(make_df):
    df_na = make_df(DATA)
    with pytest.raises(ValueError):
        imputer = RandomSampleImputer(seed="observation", random_state="arbitrary")
        imputer.fit(df_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_variables_cast_as_category(make_df):
    df_na = make_df(DATA)
    if make_df is pd.DataFrame:
        df_na["City"] = df_na["City"].astype("category")
    else:
        df_na = df_na.with_columns(pl.col("City").cast(pl.Categorical))

    imputer = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed = imputer.fit_transform(df_na)

    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 5
    assert _null_count(X_transformed, "City") == 0
    assert set(_values(X_transformed, "City")) <= _pool(df_na, "City")
