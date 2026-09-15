# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import re

import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.imputation import RandomSampleImputer
from feature_engine.imputation.random_sample import _hash_seeds
from tests.backend_helpers import frame_to_dict, null_count


# init parameters
@pytest.mark.parametrize(
    "seed", ["arbitrary", "both", 1, None, ("general",), ["observation"]]
)
def test_error_if_seed_not_permitted_value(seed):
    msg = f"seed takes only values 'general' or 'observation'. Got {seed} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        RandomSampleImputer(seed=seed)


@pytest.mark.parametrize("random_state", ["arbitrary", 0.5, ["Age"]])
def test_error_if_random_state_not_integer_when_seed_is_general(random_state):
    msg = (
        "if seed == 'general' then random_state must take an integer. "
        f"Got {random_state} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        RandomSampleImputer(seed="general", random_state=random_state)


@pytest.mark.parametrize("random_state", [None, [], ""])
def test_error_if_random_state_is_empty_when_seed_is_observation(random_state):
    msg = (
        "if seed == 'observation' the random state must take the name of one "
        "or more variables which will be used to seed the imputer. "
        f"Got {random_state} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        RandomSampleImputer(seed="observation", random_state=random_state)


@pytest.mark.parametrize(
    "random_state, seed",
    [
        (None, "general"),
        (5, "general"),
        ("Age", "observation"),
        (["Age", "Marks"], "observation"),
    ],
)
def test_init_param_assignment(random_state, seed):
    imputer = RandomSampleImputer(random_state=random_state, seed=seed)
    assert imputer.random_state == random_state
    assert imputer.seed == seed


# fit and transform
def test_hash_seeds():
    values = np.array(
        [
            [25, 0.7],
            [25.0, 0.7],
            [0.0, 0.7],
            [np.nan, 0.7],
            [-0.0, 0.7],
            [-30.0, 1e20],
        ]
    )
    seeds = _hash_seeds(values)

    # same values, same seed: ints and floats are equal, nan and -0.0 count as 0
    assert seeds[0] == seeds[1]
    assert seeds[2] == seeds[3] == seeds[4]
    assert seeds[0] != seeds[2]
    # negative and large values give valid numpy seeds
    assert all(0 <= seed < 2**32 for seed in seeds)
    # the seed must not change between sessions or releases
    assert _hash_seeds(np.array([[25.0, 0.7]]))[0] == 2067629302


def test_general_seed_plus_automatically_select_variables(make_df, data_na):
    df_na = make_df(data_na)
    imputer = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed = imputer.fit_transform(df_na)

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

    # pandas and polars draw different values for the same seed, so we only check
    # that the same seed on the same backend gives the same result.
    imputer2 = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed2 = imputer2.fit_transform(df_na)
    assert frame_to_dict(X_transformed) == frame_to_dict(X_transformed2)


def test_pandas_general_seed_reproduces_historic_values(df_na):
    # pandas only: with a fixed seed, pandas must return the same values as before
    # the narwhals migration. polars uses a different random number generator.
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


@pytest.mark.parametrize("random_state", [["Marks", "Age"], "Age"])
def test_seed_per_observation(make_df, data_na, random_state):
    seed_vars = [random_state] if isinstance(random_state, str) else random_state
    data = _data_without_na_in(data_na, seed_vars)
    df_na = make_df(data)

    imputer = RandomSampleImputer(
        variables=["City", "Studies"],
        random_state=random_state,
        seed="observation",
    )
    X_transformed = imputer.fit_transform(df_na)

    # fit() turns a single seeding variable name into a list
    assert imputer.random_state == seed_vars
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
    )
    X_transformed2 = imputer2.fit_transform(df_na)
    assert frame_to_dict(X_transformed) == frame_to_dict(X_transformed2)


DATA_SEED_TRAIN = {
    "City": ["London", "Manchester", "Bristol", "Leeds", "York", "Bath", "Hull"],
    "Age": [20.0, 21.0, 19.0, 23.0, 40.0, 41.0, 37.0],
    "Marks": [0.9, 0.8, 0.7, 0.3, 0.6, 0.8, 0.5],
}
# rows 0 and 3 have identical seeding values and City missing
DATA_SEED_TEST = {
    "City": [None, "Leeds", None, None, None],
    "Age": [25.0, 30.0, 40.0, 25.0, 33.0],
    "Marks": [0.7, 0.4, 0.6, 0.7, 0.2],
}


def test_seed_per_observation_imputes_identical_rows_equally(make_df):
    imputer = RandomSampleImputer(
        variables=["City"], random_state=["Age", "Marks"], seed="observation"
    )
    imputer.fit(make_df(DATA_SEED_TRAIN))
    X_transformed = imputer.transform(make_df(DATA_SEED_TEST))

    city = frame_to_dict(X_transformed)["City"]
    assert city[0] == city[3]


def test_seed_per_observation_does_not_depend_on_row_position(make_df):
    imputer = RandomSampleImputer(
        variables=["City"], random_state=["Age", "Marks"], seed="observation"
    )
    imputer.fit(make_df(DATA_SEED_TRAIN))
    X = make_df(DATA_SEED_TEST)
    expected = frame_to_dict(imputer.transform(X))["City"]

    # same rows in reverse order
    X_reversed = make_df({k: v[::-1] for k, v in DATA_SEED_TEST.items()})
    reversed_city = frame_to_dict(imputer.transform(X_reversed))["City"]
    assert reversed_city == expected[::-1]

    # each row imputed on its own
    for i in range(len(expected)):
        row_city = frame_to_dict(imputer.transform(X[i:i + 1]))["City"]
        assert row_city == [expected[i]]


def test_seed_per_observation_with_negative_and_large_seeding_values(make_df):
    imputer = RandomSampleImputer(
        variables=["City"], random_state=["Age", "Marks"], seed="observation"
    )
    imputer.fit(make_df(DATA_SEED_TRAIN))
    X = make_df(
        {
            "City": [None, None, "Leeds"],
            "Age": [-30.0, 1e20, 20.0],
            "Marks": [0.1, 1e20, 0.9],
        }
    )
    X_transformed = imputer.transform(X)

    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "City") == 0
    assert set(frame_to_dict(X_transformed)["City"]) <= set(DATA_SEED_TRAIN["City"])


def test_seed_per_observation_uses_values_before_imputation_with_missing_as_zero(
    make_df,
):
    # Age is imputed and also seeds City: row 0 (Age missing) must seed like
    # row 1 (Age 0), not with its imputed Age.
    imputer = RandomSampleImputer(
        variables=["Age", "City"], random_state=["Age", "Marks"], seed="observation"
    )
    imputer.fit(make_df(DATA_SEED_TRAIN))
    X = make_df(
        {
            "City": [None, None, "Leeds"],
            "Age": [None, 0.0, 30.0],
            "Marks": [0.7, 0.7, 0.4],
        }
    )
    X_transformed = imputer.transform(X)

    result = frame_to_dict(X_transformed)
    assert null_count(X_transformed, "Age") == 0
    assert result["City"][0] == result["City"][1]


def test_seed_per_observation_with_duplicated_index():
    # pandas only: polars has no index
    imputer = RandomSampleImputer(
        variables=["City"], random_state=["Age", "Marks"], seed="observation"
    )
    imputer.fit(pd.DataFrame(DATA_SEED_TRAIN))
    X = pd.DataFrame(DATA_SEED_TEST)
    expected = frame_to_dict(imputer.transform(X))["City"]

    X.index = [0, 0, 1, 1, 2]
    assert frame_to_dict(imputer.transform(X))["City"] == expected


def test_error_if_random_state_variables_not_in_dataframe(make_df, data_na):
    imputer = RandomSampleImputer(seed="observation", random_state="arbitrary")
    msg = (
        "There are variables assigned as random state which are not part "
        "of the training dataframe. Got arbitrary instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
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
