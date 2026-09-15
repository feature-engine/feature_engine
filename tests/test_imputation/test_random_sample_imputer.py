# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import pandas as pd
import polars as pl
import pytest

from feature_engine.imputation import RandomSampleImputer
from feature_engine.imputation.random_sample import _define_seed
from tests.backend_helpers import null_count, frame_to_dict


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


def test_general_seed_plus_automatically_select_variables(make_df, data_na):
    df_na = make_df(data_na)
    imputer = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed = imputer.fit_transform(df_na)

    # test init params
    assert imputer.variables is None
    assert imputer.random_state == 5
    assert imputer.seed == "general"

    # test fit attrs
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 5
    assert frame_to_dict(imputer.X_) == frame_to_dict(df_na)

    # no missing data left in any imputed variable, and every value used to
    # fill NA came from the training data itself
    assert isinstance(X_transformed, make_df)
    result = frame_to_dict(X_transformed)
    for col in imputer.variables_:
        assert null_count(X_transformed, col) == 0
        assert set(result[col]) <= {v for v in data_na[col] if v is not None}

    # pandas' and narwhals/polars' sample() use different RNGs, so a fixed
    # seed does not draw the same values across backends - only same seed +
    # same backend is a reproducibility guarantee. Verify that guarantee.
    imputer2 = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed2 = imputer2.fit_transform(df_na)
    assert frame_to_dict(X_transformed) == frame_to_dict(X_transformed2)


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


def _data_without_na_in(data, columns):
    # the variables used as seed should not have missing data
    data = dict(data)
    for col in columns:
        data[col] = [v if v is not None else 1 for v in data[col]]
    return data


@pytest.mark.parametrize(
    "random_state,seeding_method",
    [(["Marks", "Age"], "add"), (["Marks", "Age"], "multiply"), ("Age", "add")],
)
def test_seed_per_observation(make_df, data_na, random_state, seeding_method):
    seed_vars = [random_state] if isinstance(random_state, str) else random_state
    data = _data_without_na_in(data_na, seed_vars)
    df_na = make_df(data)

    imputer = RandomSampleImputer(
        variables=["City", "Studies"],
        random_state=random_state,
        seed="observation",
        seeding_method=seeding_method,
    )
    X_transformed = imputer.fit_transform(df_na)

    assert imputer.variables == ["City", "Studies"]
    assert imputer.random_state == seed_vars
    assert imputer.seed == "observation"
    assert isinstance(X_transformed, make_df)
    result = frame_to_dict(X_transformed)
    for col in ["City", "Studies"]:
        assert frame_to_dict(imputer.X_)[col] == data[col]
        assert null_count(X_transformed, col) == 0
        assert set(result[col]) <= {v for v in data[col] if v is not None}
    # variables not selected for imputation are untouched
    assert result["Age"] == data["Age"]

    # same seed, same backend -> same result
    imputer2 = RandomSampleImputer(
        variables=["City", "Studies"],
        random_state=random_state,
        seed="observation",
        seeding_method=seeding_method,
    )
    X_transformed2 = imputer2.fit_transform(df_na)
    assert frame_to_dict(X_transformed) == frame_to_dict(X_transformed2)


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


def test_error_if_random_state_is_string(make_df, data_na):
    imputer = RandomSampleImputer(seed="observation", random_state="arbitrary")
    with pytest.raises(ValueError):
        imputer.fit(make_df(data_na))


def test_variables_cast_as_category(make_df, data_na):
    df_na = make_df(data_na)
    if make_df is pd.DataFrame:
        df_na["City"] = df_na["City"].astype("category")
    else:
        df_na = df_na.with_columns(pl.col("City").cast(pl.Categorical))

    imputer = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed = imputer.fit_transform(df_na)

    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 5
    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "City") == 0
    city_pool = {v for v in data_na["City"] if v is not None}
    assert set(frame_to_dict(X_transformed)["City"]) <= city_pool
