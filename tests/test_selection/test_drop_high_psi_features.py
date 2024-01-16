import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_engine.selection import DropHighPSIFeatures


@pytest.fixture(scope="module")
def df():
    # create array with 4 correlated features and 2 independent ones
    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # transform array into pandas df
    colnames = ["var_" + str(i) for i in range(6)]
    X = pd.DataFrame(X, columns=colnames)

    # Add a categorical column that does not drift
    X["cat_1"] = ["A", "B"] * int(X.shape[0] / 2)

    # Add drifted features that will be dropped during transformation.
    X["drift_1"] = [number for number in range(X.shape[0])]
    X["drift_2"] = [number / 2 for number in range(X.shape[0])]
    X["drift_cat_1"] = ["A" for _ in range(int(X.shape[0] / 2))] + [
        "B" for _ in range(int(X.shape[0] / 2))
    ]
    X["drift_cat_1"] = X["drift_cat_1"].astype("category")

    return X


@pytest.fixture(scope="module")
def df_mixed_types():
    df = pd.DataFrame(
        {
            "A": [it for it in range(0, 20)],
            "B": [1, 2, 2, 1] * 5,
            "C": ["A", "B", "D", "D"] * 5,
            "time": [datetime(2019, 1, it + 1) for it in range(20)],
        }
    )

    return df


EXPECTED_PSI_NUM = {
    "var_0": 0.043828484052281,
    "var_1": 0.040929870747665395,
    "var_2": 0.04330418495156895,
    "var_3": 0.03773286532548153,
    "var_4": 0.05047388515663041,
    "var_5": 0.014717735595712466,
    "drift_1": 8.283089355027482,
    "drift_2": 8.283089355027482,
}

EXPECTED_PSI_ALL = {
    "var_0": 0.043828484052281,
    "var_1": 0.040929870747665395,
    "var_2": 0.04330418495156895,
    "var_3": 0.03773286532548153,
    "var_4": 0.05047388515663041,
    "var_5": 0.014717735595712466,
    "drift_1": 8.283089355027482,
    "drift_2": 8.283089355027482,
    "cat_1": 0.0,
    "drift_cat_1": 18.41883867587797,
}

EXPECTED_PSI_NUM_FEW = {
    "var_2": 0.04330418495156895,
    "var_3": 0.03773286532548153,
    "drift_1": 8.283089355027482,
    "drift_2": 8.283089355027482,
}

EXPECTED_PSI_MIXED = {
    "var_0": 0.043828484052281,
    "drift_1": 8.283089355027482,
    "drift_cat_1": 18.41883867587797,
}

EXPECTED_PSI_CAT_FEW = {
    "cat_1": 0.0,
    "drift_cat_1": 18.41883867587797,
}

EXPECTED_PSI_STRING = {"var_0": 0.043828484052281}

_input_output = [
    (None, EXPECTED_PSI_NUM),
    ("all", EXPECTED_PSI_ALL),
    (["var_2", "var_3", "drift_1", "drift_2"], EXPECTED_PSI_NUM_FEW),
    (["cat_1", "drift_cat_1"], EXPECTED_PSI_CAT_FEW),
    (["var_0", "drift_1", "drift_cat_1"], EXPECTED_PSI_MIXED),
    ("var_0", EXPECTED_PSI_STRING),
]


# ====  test  main functionality of the class ====
@pytest.mark.parametrize("variables, expected_psi", _input_output)
def test_fit_attributes(variables, expected_psi, df):
    """Check the value of the fit attributes.
    The expected PSI values used in the assertion were determined using
    the Probatus package.
    ```
    from probatus.stat_tests import AutoDist
    psi_calculator = AutoDist(statistical_tests=["PSI"],
                    binning_strategies="QuantileBucketer",
                    bin_count=10)
    train_df = data.iloc[0:500,:]
    test_df = data.iloc[500:, :]
    psi = psi_calculator.compute(train_df, test_df)
    ```
    """
    transformer = DropHighPSIFeatures(variables=variables)
    dft = transformer.fit_transform(df)

    assert transformer.variables_ == list(expected_psi.keys())
    assert transformer.psi_values_ == pytest.approx(expected_psi, 12)
    assert transformer.features_to_drop_ == [
        var for var in expected_psi.keys() if "drift" in var
    ]
    pd.testing.assert_frame_equal(dft, df.drop(transformer.features_to_drop_, axis=1))


def test_auto_threshold_calculation():
    """Check the results of 'auto' threshold calculation"""
    transformer = DropHighPSIFeatures(threshold="auto", p_value=0.001, bins=10)
    assert math.isclose(
        transformer._calculate_auto_threshold(N=500, M=500, bins=10),
        0.11150865948502628,
    )
    transformer = DropHighPSIFeatures(threshold="auto", p_value=0.05, bins=32)
    assert math.isclose(
        transformer._calculate_auto_threshold(N=1000, M=1500, bins=32),
        0.07497557213394188,
    )
    transformer = DropHighPSIFeatures(threshold="auto", p_value=0.01, bins=42)
    assert math.isclose(
        transformer._calculate_auto_threshold(N=777, M=666, bins=42),
        0.18111345503169146,
    )


@pytest.mark.parametrize("variables, expected_psi", _input_output)
def test_fit_attributes_with_autothreshold(variables, expected_psi, df):
    """Check the value of the fit attributes.
    The expected PSI values used in the assertion were determined using
    the Probatus package.
    ```
    from probatus.stat_tests import AutoDist
    psi_calculator = AutoDist(statistical_tests=["PSI"],
                    binning_strategies="QuantileBucketer",
                    bin_count=10)
    train_df = data.iloc[0:500,:]
    test_df = data.iloc[500:, :]
    psi = psi_calculator.compute(train_df, test_df)
    ```
    """
    transformer = DropHighPSIFeatures(threshold="auto", variables=variables, bins=10)
    transformer.fit(df)

    assert transformer.psi_values_ == pytest.approx(expected_psi, 12)
    assert transformer.features_to_drop_ == [
        var for var in expected_psi.keys() if "drift" in var
    ]


def test_calculation_when_strategy_equal_width(df):
    transformer = DropHighPSIFeatures(strategy="equal_width")
    transformer.fit(df)

    expected = {
        "var_0": 0.014858665472468786,
        "var_1": 0.04514737836588022,
        "var_2": 0.03431479397506742,
        "var_3": 0.04298209189840294,
        "var_4": 0.02385796430263416,
        "var_5": 0.046809664317794444,
        "drift_1": 8.283089355027482,
        "drift_2": 8.283089355027482,
    }

    assert transformer.psi_values_ == pytest.approx(expected, 12)
    assert transformer.features_to_drop_ == [
        var for var in expected.keys() if "drift" in var
    ]


# ================ test init parameters =================

# Define two dictionaries with arguments: one with default values and
# one with arbitrary values.
default_dict = {
    "split_col": None,
    "split_frac": 0.5,
    "split_distinct": False,
    "cut_off": None,
    "switch": False,
    "threshold": 0.25,
    "bins": 10,
    "strategy": "equal_frequency",
    "min_pct_empty_bins": 0.0001,
    "missing_values": "raise",
    "variables": None,
    "p_value": 0.001,
}

args_dict = {
    "split_col": "hola",
    "split_frac": 0.6,
    "split_distinct": True,
    "cut_off": ["value_1", "value_2"],
    "switch": True,
    "threshold": 0.10,
    "bins": 5,
    "strategy": "equal_width",
    "min_pct_empty_bins": 0.1,
    "missing_values": "ignore",
    "variables": ["chau", "adios"],
    "p_value": 0.2,
}

init_dict = [(None, default_dict), (args_dict, args_dict)]


@pytest.mark.parametrize("initialize, attribute_dict", init_dict)
def test_init_default_parameters(initialize, attribute_dict):
    """Test the default param values are correctly assigned."""
    if initialize:
        transformer = DropHighPSIFeatures(**attribute_dict)
    else:
        transformer = DropHighPSIFeatures()

    for attribute, value in attribute_dict.items():
        assert getattr(transformer, attribute) == value


def test_init_value_error_is_raised():
    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_col=["hola"])

    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_col="hola", variables=["hola", "chau"])

    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_frac=0)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_frac=1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_frac=None, cut_off=None)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_distinct=1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(bins=1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(threshold=-1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(switch=1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(strategy="unknown")

    with pytest.raises(ValueError):
        DropHighPSIFeatures(min_pct_empty_bins="unknown")

    with pytest.raises(ValueError):
        DropHighPSIFeatures(min_pct_empty_bins=-1)


@pytest.mark.parametrize("p_value", ["hola", -1, 10])
def test_p_value_not_allowed(p_value):
    with pytest.raises(ValueError):
        DropHighPSIFeatures(p_value=p_value)


# ================= test fit() functionality ==================


def test_split_col_not_included_in_variables(df):
    """Check that the split column is not included among the features
    to evaluate when these are selected automatically."""
    transformer = DropHighPSIFeatures(split_col="var_3", variables=None)
    transformer.fit(df)
    assert "var_3" not in transformer.variables_
    assert "var_3" not in transformer.psi_values_.keys()

    transformer = DropHighPSIFeatures(split_col="var_3", variables="all")
    transformer.fit(df)
    assert "var_3" not in transformer.variables_
    assert "var_3" not in transformer.psi_values_.keys()

    transformer = DropHighPSIFeatures(split_col="cat_1", variables="all")
    transformer.fit(df)
    assert "cat_1" not in transformer.variables_
    assert "cat_1" not in transformer.psi_values_.keys()


def test_error_split_col_not_in_df(df):
    transformer = DropHighPSIFeatures(variables=None, split_col="var_0")
    data = df.copy()
    data = data.drop(["var_0"], axis=1)
    msg = "var_0 is not in the dataframe."
    with pytest.raises(ValueError) as record:
        transformer.fit(data)
    assert str(record.value) == msg


_input_output = [
    (["var_2", "var_3", "drift_1", "drift_2"], ["drift_1"], EXPECTED_PSI_NUM_FEW),
    (["var_0", "drift_1", "drift_cat_1"], ["drift_cat_1"], EXPECTED_PSI_MIXED),
]


@pytest.mark.parametrize("variables, variable, expected_psi", _input_output)
def test_confirm_variables(variables, variable, expected_psi, df):
    data = df.copy()
    data = data.drop(variable, axis=1)
    del expected_psi[variable[0]]

    transformer = DropHighPSIFeatures(variables=variables, confirm_variables=True)
    transformer.fit(data)

    assert transformer.variables_ == list(expected_psi.keys())
    assert transformer.psi_values_ == pytest.approx(expected_psi, 12)
    assert transformer.features_to_drop_ == [
        var for var in expected_psi.keys() if "drift" in var
    ]


def test_error_if_variables_is_none_and_no_numerical_in_df(df):
    transformer = DropHighPSIFeatures(variables=None)
    msg = (
        "No numerical variables found in this dataframe. Please check "
        "variable format with pandas dtypes."
    )
    with pytest.raises(TypeError) as record:
        transformer.fit(df[["cat_1", "drift_cat_1"]])

    assert str(record.value) == msg


def test_error_if_confirm_variables_returns_empty_list(df):
    transformer = DropHighPSIFeatures(
        variables=["cat_1", "drift_cat_1"], confirm_variables=True
    )
    data = df.copy()
    data = data.drop(["cat_1", "drift_cat_1"], axis=1)
    msg = "None of the variables in the list are present in the dataframe."
    with pytest.raises(ValueError) as record:
        transformer.fit(data)
    assert str(record.value) == msg


def test_error_if_na_in_split_col(df):
    """Test an error is raised if the split column contains missing values."""
    data = df.copy()
    data.loc[15, "var_3"] = np.nan

    transformer = DropHighPSIFeatures(split_col="var_3")

    with pytest.raises(ValueError):
        transformer.fit(data)


def test_raise_error_if_na_in_df(df):
    """Test an error is raised when missing values is set to raise."""
    data = df.copy()
    data.loc[15, "var_3"] = np.nan

    transformer = DropHighPSIFeatures(missing_values="raise")

    with pytest.raises(ValueError):
        transformer.fit(data)


def test_missing_value_ignored(df):
    """Test if PSI are computed when missing values are present in the dataframe."""
    data = df.copy()
    data.loc[15, "var_3"] = np.nan

    transformer = DropHighPSIFeatures(missing_values="ignore")
    transformed = transformer.fit_transform(data)

    assert transformer.psi_values_ == pytest.approx(EXPECTED_PSI_NUM, 12)
    assert transformer.features_to_drop_ == [
        var for var in EXPECTED_PSI_NUM if "drift" in var
    ]
    pd.testing.assert_frame_equal(
        transformed, data.drop(transformer.features_to_drop_, axis=1)
    )


def test_raise_error_if_inf_in_df(df):
    """Test an error is raised for inf when missing values is set to raise."""
    data = df.copy()
    data.loc[15, "var_3"] = np.inf

    transformer = DropHighPSIFeatures(missing_values="raise")

    with pytest.raises(ValueError):
        transformer.fit(data)


# ========= tests for _split_dataframe() fit ====

# tests for splits based on split_frac and numerical variables:

quantile_test = [(0.5, 50), (0.33, 33), (0.17, 17), (0.81, 81)]


@pytest.mark.parametrize("split_frac, expected", quantile_test)
def test_calculation_quantile(split_frac, expected):
    """Test the calculation of the quantiles using numerical values."""
    df = pd.DataFrame(
        {"A": [it for it in range(0, 101)], "B": [it for it in range(0, 101)]}
    )

    test = DropHighPSIFeatures(
        split_col="A", split_frac=split_frac, split_distinct=False
    )
    test.fit_transform(df)
    assert test.cut_off_ == expected


quantile_test_skewed = [(50, 50), (1, 30), (10, 40), (7, 80)]


@pytest.mark.parametrize("index, fraction", quantile_test_skewed)
def test_quatile_split_skewed_variables(index, fraction):
    """Test the calculation of the quantiles using numerical and skewed variables."""
    df = pd.DataFrame(
        {
            "A": [index for it in range(0, fraction + 1)]
            + [it for it in range(fraction + 1, 101)],
            "B": [it for it in range(0, 101)],
        }
    )

    test = DropHighPSIFeatures(
        split_col="A", split_frac=fraction / 100, split_distinct=False
    )
    test.fit_transform(df)

    assert test.cut_off_ == index


# tests for splits based on split_frac and categorical variables:


def test_calculation_distinct_value_categorical():
    """Test the calculation of the quantiles using distinct values when reference
    variable is categorical."""
    df = pd.DataFrame(
        {"A": [it for it in range(0, 30)], "C": ["A", "B", "C", "D", "D", "D"] * 5}
    )

    test = DropHighPSIFeatures(split_col="C", split_frac=0.5, split_distinct=False)

    test.fit_transform(df)
    assert test.cut_off_ == "C"

    test = DropHighPSIFeatures(split_col="C", split_frac=0.5, split_distinct=True)
    test.fit_transform(df)
    assert test.cut_off_ == "B"

    df = pd.DataFrame(
        {"A": [it for it in range(0, 30)], "C": ["A", "A", "A", "B", "C", "D"] * 5}
    )

    test = DropHighPSIFeatures(split_col="C", split_frac=0.5, split_distinct=False)

    test.fit_transform(df)
    assert test.cut_off_ == "A"

    test = DropHighPSIFeatures(split_col="C", split_frac=0.5, split_distinct=True)
    test.fit_transform(df)
    assert test.cut_off_ == "B"


numerical_split_distinct = [(True, [1, 2, 3], [4, 5, 6]), (False, [1], [2, 3, 4, 5, 6])]


@pytest.mark.parametrize("split_distinct, a_values, b_values", numerical_split_distinct)
def test_split_distinct_with_numerical_values(split_distinct, a_values, b_values):
    """Test the split_distinct functionality with numerical variables."""
    # Define the testing dataframe
    df = pd.DataFrame(
        {
            "ID": [1, 1, 2, 3, 1, 4, 5, 1, 1, 6],
            "numerical": [1, 1, 1, 4, 1, 4, 3, 7, 1, 3],
        }
    )
    a_expected = df[df.ID.isin(a_values)]
    b_expected = df[df.ID.isin(b_values)]
    # Run the split_dataframe method to extract the input of the PSI calculation.
    transformer = DropHighPSIFeatures(split_col="ID", split_distinct=split_distinct)
    a, b = transformer._split_dataframe(df)
    # Test if the functionality provides the expected results.
    pd.testing.assert_frame_equal(a, a_expected)
    pd.testing.assert_frame_equal(b, b_expected)


def test_calculation_df_split_with_different_variable_types(df_mixed_types):
    """Test the split of the dataframe using different type of variables."""
    results = {}
    cut_offs = {}
    for split_col in df_mixed_types.columns:
        test = DropHighPSIFeatures(split_frac=0.5, split_col=split_col, variables="all")
        test.fit_transform(df_mixed_types)
        results[split_col] = test.psi_values_
        cut_offs[split_col] = test.cut_off_

    assert results["A"] == pytest.approx({"B": 0.0, "C": 0.1621860432432657}, 12)
    assert results["B"] == pytest.approx(
        {"A": 3.0375978817052403, "C": 8.515489752777954}, 12
    )
    assert results["C"] == pytest.approx({"A": 2.27819841127893, "B": 0.0}, 12)
    assert results["time"] == pytest.approx(
        {"A": 8.283089355027482, "B": 0.0, "C": 0.1621860432432657}, 12
    )

    expected_cut_offs = {
        "A": 9.5,
        "B": 1.5,
        "C": "B",
        "time": np.datetime64(datetime(2019, 1, 10)),
    }

    assert cut_offs == expected_cut_offs

    # Test when no dataframe with mixed data types when no split_col is provided.
    test = DropHighPSIFeatures(split_frac=0.5, variables="all")
    test.fit_transform(df_mixed_types)
    assert test.psi_values_ == pytest.approx(
        {"A": 8.283089355027482, "B": 0.0, "C": 0.1621860432432657}, 12
    )


# =========== tests for user entered cut_off values ===========


type_test = [
    ("A", 14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    ("B", 1, [0, 3, 4, 7, 8, 11, 12, 15, 16, 19]),
    ("C", ["B"], [1, 5, 9, 13, 17]),
    ("C", "B", [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]),
    ("time", datetime(2019, 1, 4), [0, 1, 2, 3]),
]


@pytest.mark.parametrize("col, cut_off, expected", type_test)
def test_split_using_cut_off(col, cut_off, expected, df_mixed_types):
    """Test the cut off for different data types."""
    test = DropHighPSIFeatures(split_col=col, cut_off=cut_off)
    a, b = test._split_dataframe(df_mixed_types)

    pd.testing.assert_frame_equal(a, df_mixed_types.loc[expected])
    pd.testing.assert_frame_equal(
        b, df_mixed_types.loc[~df_mixed_types.index.isin(expected)]
    )


split_distinct_test = [
    ("A", [number for number in range(0, 100)]),
    (
        "B",
        (
            [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40]
            + [41, 42, 50, 51, 52, 60, 61, 62, 70, 71, 72, 80, 81]
            + [82, 90, 91, 92, 100, 101, 102, 110, 111, 112, 120, 121, 122]
            + [130, 131, 132, 140, 141, 142, 150, 151, 152, 160, 161, 162, 170]
            + [171, 172, 180, 181, 182, 190, 191, 192]
        ),
    ),
    (
        "C",
        (
            [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40]
            + [41, 42, 50, 51, 52, 60, 61, 62, 70, 71, 72, 80, 81]
            + [82, 90, 91, 92, 100, 101, 102, 110, 111, 112, 120, 121, 122]
            + [130, 131, 132, 140, 141, 142, 150, 151, 152, 160, 161, 162, 170]
            + [171, 172, 180, 181, 182, 190, 191, 192]
        ),
    ),
    (
        "time",
        (
            [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26]
            + [27, 30, 31, 32, 35, 36, 37, 40, 41, 42, 45, 46, 47, 50, 51, 52, 55]
            + [56, 57, 60, 61, 62, 65, 66, 67, 70, 71, 72, 75, 76, 77, 80, 81, 82]
            + [85, 86, 87, 90, 91, 92, 95, 96, 97]
        ),
    ),
]


@pytest.mark.parametrize("col, expected_index", split_distinct_test)
def test_split_distinct(col, expected_index):
    """Test the cut off for different data types.
    For columns B, C and time we have 6 distinct values, 5 appearing 20 times and
    1 appearing 100 times. A 50% split based on the number of values will result
    in 2 groups of 3. One has 60 appearances (in total) and the other has 140.
    """
    data = pd.DataFrame(
        {
            "A": [it for it in range(0, 200)],
            "B": [1, 2, 3, 4, 5, 6, 6, 6, 6, 6] * 20,
            "C": ["A", "B", "C", "D", "E", "F", "F", "F", "F", "F"] * 20,
            "time": [date(2019, 1, it + 1) for it in range(5)] * 20
            + [date(2019, 1, 31)] * 100,
        }
    )
    test = DropHighPSIFeatures(split_col=col, split_distinct=True)
    a, b = test._split_dataframe(data)

    pd.testing.assert_frame_equal(a, data.loc[expected_index])
    pd.testing.assert_frame_equal(b, data.loc[~data.index.isin(expected_index)])


cut_off_list_test = [
    ("A", [1, 2, 10, 11, 16]),
    ("B", [2]),
    ("C", ["B", "D"]),
    ("time", [date(2019, 1, day) for day in [1, 2, 5, 7, 12, 15, 18]]),
]


@pytest.mark.parametrize("col, cut_off_list", cut_off_list_test)
def test_split_by_list(df_mixed_types, col, cut_off_list):
    """Test elements a correctly selected when cut_off is a list."""
    test = DropHighPSIFeatures(split_col=col, cut_off=cut_off_list, bins=3)
    a, b = test._split_dataframe(df_mixed_types)

    pd.testing.assert_frame_equal(
        a, df_mixed_types[df_mixed_types[col].isin(cut_off_list)]
    )
    pd.testing.assert_frame_equal(
        b, df_mixed_types[~df_mixed_types[col].isin(cut_off_list)]
    )


# Tests for split on index on shuffled dataframe
def test_split_shuffled_df_default(df):
    """Test the default parameters when the index is shuffled."""
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1)
    test = DropHighPSIFeatures()
    base, test = test._split_dataframe(df_shuffled)

    # The base dataframe should contain indexes from 0 to 499
    set(base.index) == {_ for _ in range(0, 500)}

    # The test dataframe should contain indexes from 500 to 999
    set(test.index) == {_ for _ in range(500, 999)}


def test_split_shuffled_df_split_frac(df):
    """Test split_frac when the index is shuffled."""
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1)
    test = DropHighPSIFeatures(split_frac=0.6)
    base, test = test._split_dataframe(df_shuffled)

    # The base dataframe should contain indexes from 0 to 599
    set(base.index) == {_ for _ in range(0, 600)}

    # The test dataframe should contain indexes from 600 to 999
    set(test.index) == {_ for _ in range(600, 999)}


def test_split_shuffled_df_cut_off(df):
    """Test the cut_off when the index is shuffled."""
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1)
    test = DropHighPSIFeatures(cut_off=250)
    base, test = test._split_dataframe(df_shuffled)

    # The base dataframe should contain indexes from 0 to 250
    set(base.index) == {_ for _ in range(0, 251)}

    # The test dataframe should contain indexes from 251 to 999
    set(test.index) == {_ for _ in range(251, 999)}


# ===== end of tests for _split_dataframe() =======

# ==== more tests for fit functionality ============


def test_switch():
    """Test the functionality to switch the basis."""

    df_a = pd.DataFrame(
        {
            "a": [1.0, 2, 3, 1],
            "b": [1.0, 2, 3, 4],
            "c": [1, 2, 3, 4],
            "d": [1.7, 4.7, 6.6, 7.8],
        }
    )

    df_b = pd.DataFrame(
        {
            "a": [4.0, 3, 5, 1],
            "b": [11.0, 1, 2, 4],
            "c": [4, 2, 2, 4],
            "d": [4.7, 4.7, 7.6, 7.8],
        }
    )

    df_order = pd.concat([df_a, df_b]).reset_index(drop=True)
    df_reverse = pd.concat([df_b, df_a]).reset_index(drop=True)

    case = DropHighPSIFeatures(
        split_frac=0.5, bins=3, switch=False, min_pct_empty_bins=0.001
    )
    case.fit(df_order)

    switch_case = DropHighPSIFeatures(
        split_frac=0.5, bins=3, switch=True, min_pct_empty_bins=0.001
    )
    switch_case.fit(df_reverse)

    assert case.psi_values_ == switch_case.psi_values_


def test_observation_frequency_per_bin():
    """Test empty bins are populated by a tiny amount."""
    a = pd.DataFrame({"A": [1, 2, 4]})
    b = pd.DataFrame({"A": [1, 2, 3]})
    transformer = DropHighPSIFeatures()
    a_bins, b_bins = transformer._observation_frequency_per_bin(a, b)

    expected_a_bins = pd.Series([0.3333333, 0.333333, 0.0001, 0.333333])
    expected_b_bins = pd.Series([0.3333333, 0.333333, 0.333333, 0.0001])

    pd.testing.assert_series_equal(
        a_bins.reset_index(drop=True), expected_a_bins, check_names=False
    )
    pd.testing.assert_series_equal(
        b_bins.reset_index(drop=True), expected_b_bins, check_names=False
    )


def test_transform_feature_to_drop_not_present(df):
    """Test transform when the feature to drop in not in the dataframe."""
    test = DropHighPSIFeatures()
    test.fit(df)

    # Define new dataframe with additional column
    data = df.copy()
    data["A"] = [1] * 1000
    # Remove one of the feature to drop
    data = data.drop("drift_1", axis=1)

    with pytest.raises(KeyError):
        test.transform(data)


def test_transform_different_number_of_columns(df):
    """Test transform on df with different number of features to train set."""
    test = DropHighPSIFeatures()
    test.fit(df)

    # Define new dataframe with additional column
    data = df.copy()
    data["A"] = [1] * 1000

    with pytest.raises(ValueError):
        test.transform(data)
