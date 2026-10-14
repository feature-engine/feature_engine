# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import pytest

from feature_engine.imputation import RandomSampleImputer
from feature_engine.imputation.random_sample import _define_seed


def test_define_seed(df_vartypes):
    assert _define_seed(df_vartypes, 0, ["Age", "Marks"], how="add") == 21
    assert _define_seed(df_vartypes, 0, ["Age", "Marks"], how="multiply") == 18
    assert _define_seed(df_vartypes, 2, ["Age", "Marks"], how="add") == 20
    assert _define_seed(df_vartypes, 2, ["Age", "Marks"], how="multiply") == 13
    assert _define_seed(df_vartypes, 1, ["Age"], how="add") == 21
    assert _define_seed(df_vartypes, 3, ["Marks"], how="multiply") == 1


def test_general_seed_plus_automatically_select_variables(df_na):
    # set up transformer
    imputer = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed = imputer.fit_transform(df_na)

    # expected output:
    # fillna based on seed used (found experimenting on Jupyter notebook)
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

    # test init params
    assert imputer.variables is None
    assert imputer.random_state == 5
    assert imputer.seed == "general"

    # test fit attr
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks", "dob"]
    assert imputer.n_features_in_ == 6
    pd.testing.assert_frame_equal(imputer.X_, df_na)

    # test transform output
    pd.testing.assert_frame_equal(X_transformed, ref, check_dtype=False)


def test_seed_per_observation_and_multiple_variables_in_random_state(df_na):
    # test case 2: imputer seed per observation using multiple variables to determine
    # the random_state
    # Note the variables used as seed should not have missing data, this I fill
    df_na = df_na.copy()
    df_na[["Marks", "Age"]] = df_na[["Marks", "Age"]].fillna(1)

    imputer = RandomSampleImputer(
        variables=["City", "Studies"], random_state=["Marks", "Age"], seed="observation"
    )

    X_transformed = imputer.fit_transform(df_na)

    # expected output
    ref = {
        "Name": ["tom", "nick", "krish", np.nan, "peter", np.nan, "fred", "sam"],
        "City": [
            "London",
            "Manchester",
            "London",
            "London",
            "London",
            "London",
            "Bristol",
            "Manchester",
        ],
        "Studies": [
            "Bachelor",
            "Bachelor",
            "PhD",
            "Bachelor",
            "Bachelor",
            "PhD",
            "None",
            "Masters",
        ],
        "Age": [20, 21, 19, np.nan, 23, 40, 41, 37],
        "Marks": [0.9, 0.8, 0.7, np.nan, 0.3, np.nan, 0.8, 0.6],
        "dob": pd.date_range("2020-02-24", periods=8, freq="min"),
    }
    ref = pd.DataFrame(ref)

    assert imputer.variables == ["City", "Studies"]
    assert imputer.random_state == ["Marks", "Age"]
    assert imputer.seed == "observation"
    pd.testing.assert_frame_equal(
        imputer.X_[["City", "Studies"]], df_na[["City", "Studies"]]
    )

    pd.testing.assert_frame_equal(
        X_transformed[["City", "Studies"]], ref[["City", "Studies"]]
    )


def test_seed_per_observation_plus_product_of_seeding_variables(df_na):
    # test case 3: observation seed, 2 variables as seed, product of seed variables
    # need to fill variables used as seed
    df_na = df_na.copy()
    df_na[["Marks", "Age"]] = df_na[["Marks", "Age"]].fillna(1)

    imputer = RandomSampleImputer(
        variables=["City", "Studies"],
        random_state=["Marks", "Age"],
        seed="observation",
        seeding_method="multiply",
    )

    X_transformed = imputer.fit_transform(df_na)

    # expected output
    ref = {
        "Name": ["tom", "nick", "krish", np.nan, "peter", np.nan, "fred", "sam"],
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
            "Bachelor",
            "Masters",
            "Bachelor",
            "PhD",
            "None",
            "Masters",
        ],
        "Age": [20, 21, 19, np.nan, 23, 40, 41, 37],
        "Marks": [0.9, 0.8, 0.7, np.nan, 0.3, np.nan, 0.8, 0.6],
        "dob": pd.date_range("2020-02-24", periods=8, freq="min"),
    }
    ref = pd.DataFrame(ref)

    assert imputer.variables == ["City", "Studies"]
    assert imputer.random_state == ["Marks", "Age"]
    assert imputer.seed == "observation"

    pd.testing.assert_frame_equal(
        imputer.X_[["City", "Studies"]], df_na[["City", "Studies"]]
    )

    pd.testing.assert_frame_equal(
        X_transformed[["City", "Studies"]],
        ref[["City", "Studies"]],
        check_dtype=False,
    )


def test_seed_per_observation_with_only_1_variable_as_seed(df_na):
    # test case 4: observation seed, only variable indicated as seed, method: addition
    # Note the variable used as seed should not have missing data
    df_na = df_na.copy()
    df_na["Age"] = df_na["Age"].fillna(1)

    imputer = RandomSampleImputer(
        variables=["City", "Studies"], random_state="Age", seed="observation"
    )

    X_transformed = imputer.fit_transform(df_na)

    # expected output
    ref = {
        "Name": ["tom", "nick", "krish", np.nan, "peter", np.nan, "fred", "sam"],
        "City": [
            "London",
            "Manchester",
            "Manchester",
            "Manchester",
            "London",
            "London",
            "Bristol",
            "Manchester",
        ],
        "Studies": [
            "Bachelor",
            "Bachelor",
            "Masters",
            "Masters",
            "Bachelor",
            "PhD",
            "None",
            "Masters",
        ],
        "Age": [20, 21, 19, np.nan, 23, 40, 41, 37],
        "Marks": [0.9, 0.8, 0.7, np.nan, 0.3, np.nan, 0.8, 0.6],
        "dob": pd.date_range("2020-02-24", periods=8, freq="min"),
    }
    ref = pd.DataFrame(ref)

    assert imputer.random_state == ["Age"]

    pd.testing.assert_frame_equal(
        imputer.X_[["City", "Studies"]], df_na[["City", "Studies"]]
    )

    pd.testing.assert_frame_equal(
        X_transformed[["City", "Studies"]],
        ref[["City", "Studies"]],
        check_dtype=False,
    )


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


def test_error_if_random_state_is_string(df_na):
    with pytest.raises(ValueError):
        imputer = RandomSampleImputer(seed="observation", random_state="arbitrary")
        imputer.fit(df_na)


def test_variables_cast_as_category(df_na):

    df_na = df_na.copy()
    df_na["City"] = df_na["City"].astype("category")

    # set up transformer
    imputer = RandomSampleImputer(variables=None, random_state=5, seed="general")
    X_transformed = imputer.fit_transform(df_na)

    # expected output:
    # fillna based on seed used (found experimenting on Jupyter notebook)
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
    ref["City"] = ref["City"].astype("category")

    # test fit attr
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks", "dob"]
    assert imputer.n_features_in_ == 6
    pd.testing.assert_frame_equal(imputer.X_, df_na)

    # test transform output
    pd.testing.assert_frame_equal(X_transformed, ref, check_dtype=False)


@pytest.mark.parametrize("seed_column", [0, 2, "seed"])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("seeding_method", ["add", "multiply"])
def test_observation_seed_column_label(seed_column, as_list, seeding_method):
    X = pd.DataFrame([[1, None], [2, 1.0], [3, 2.0], [4, None]])
    X = X.rename(columns={0: seed_column})
    X.index = [10, 20, 30, 40]
    original = X.copy()
    expected = X.copy()
    expected[1] = [2.0, 1.0, 2.0, 1.0]
    random_state = [seed_column] if as_list else seed_column
    imputer = RandomSampleImputer(
        variables=[1],
        seed="observation",
        random_state=random_state,
        seeding_method=seeding_method,
    )

    result = imputer.fit_transform(X)

    assert imputer.random_state == [seed_column]
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(imputer.transform(X), expected)
    pd.testing.assert_frame_equal(X, original)
    pd.testing.assert_frame_equal(imputer.X_, original[[1]])


@pytest.mark.parametrize("random_state", [None, [], ""])
def test_empty_observation_seed_still_raises(random_state):
    with pytest.raises(ValueError, match="the random state must take the name"):
        RandomSampleImputer(seed="observation", random_state=random_state)


def test_general_seed_zero():
    X = pd.DataFrame([[1, None], [2, 1.0], [3, 2.0], [4, None]])
    expected = X.copy()
    expected[1] = [1.0, 1.0, 2.0, 2.0]
    imputer = RandomSampleImputer(variables=[1], seed="general", random_state=0)

    pd.testing.assert_frame_equal(imputer.fit_transform(X), expected)
