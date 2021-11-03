from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

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

    # Add drifted features that will be dropped during transformation.
    X["drift_1"] = [number for number in range(X.shape[0])]
    X["drift_2"] = [number / 2 for number in range(X.shape[0])]

    return X


@pytest.fixture(scope="module")
def df_mixed_types():
    df = pd.DataFrame(
        {
            "A": [it for it in range(0, 20)],
            "B": [1, 2, 2, 1] * 5,
            "C": ["A", "B", "D", "D"] * 5,
            "time": [date(2019, 1, it + 1) for it in range(20)],
        }
    )

    return df


# ====  test  main functionality of the class ====
def test_fit_attributes(df):
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
    transformer = DropHighPSIFeatures()
    transformer.fit_transform(df)

    expected_psi = {
        "var_0": 0.043828484052281,
        "var_1": 0.040929870747665395,
        "var_2": 0.04330418495156895,
        "var_3": 0.03773286532548153,
        "var_4": 0.05047388515663041,
        "var_5": 0.014717735595712466,
        "drift_1": 8.283089355027482,
        "drift_2": 8.283089355027482,
    }

    assert transformer.variables_ == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "drift_1",
        "drift_2",
    ]
    assert transformer.psi_values_ == pytest.approx(expected_psi, 12)
    assert transformer.features_to_drop_ == ["drift_1", "drift_2"]
    assert transformer.n_features_in_ == 8


# ================ test init parameters =================


def test_init_default_parameters():
    # TODO: merge this test and the following into 1, using parametrize
    # and passing all allowed values to each parameter, ie, split col can
    # take None, string, should test both.
    """Test the default param values are correctly assigned."""
    transformer = DropHighPSIFeatures()

    assert transformer.split_col is None
    assert transformer.split_frac == 0.5
    assert transformer.split_distinct is False
    assert transformer.cut_off is None
    assert transformer.switch is False
    assert transformer.threshold == 0.25
    assert transformer.bins == 10
    assert transformer.strategy == "equal_frequency"
    assert transformer.min_pct_empty_bins == 0.0001
    assert transformer.missing_values == "raise"
    assert transformer.variables is None


def test_init_alternative_params():
    # TODO: merge with previous using parametrize
    """Test user entered parameters correctly assigned"""
    transformer = DropHighPSIFeatures(
        split_col="hola",
        split_frac=0.6,
        split_distinct=True,
        cut_off=["value_1", "value_2"],
        switch=True,
        threshold=0.10,
        bins=5,
        strategy="equal_frequency",
        min_pct_empty_bins=0.1,
        missing_values="raise",
        variables=["chau", "adios"],
    )

    assert transformer.split_col == "hola"
    assert transformer.split_frac == 0.6
    assert transformer.split_distinct is True
    assert transformer.cut_off == ["value_1", "value_2"]
    assert transformer.switch is True
    assert transformer.threshold == 0.1
    assert transformer.bins == 5
    assert transformer.strategy == "equal_frequency"
    assert transformer.min_pct_empty_bins == 0.1
    assert transformer.missing_values == "raise"
    assert transformer.variables == ["chau", "adios"]


def test_init_value_error_is_raised():
    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_col=["hola"])

    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_col=["hola"], variables=["hola", "chau"])

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


# ================= test fit() functionality ==================


def test_split_col_not_included_in_variables(df):
    """Check that the split columns is not included among the features
    to evaluate when these are selected automatically."""
    transformer = DropHighPSIFeatures(split_col="var_3", variables=None)
    transformer.fit(df)

    assert transformer.variables is None
    assert "var_3" not in transformer.variables_
    assert "var_3" not in transformer.psi_values_.keys()


def test_missing_split_col(df):
    # Test an error is raised if the split column contains missing values
    data = df.copy()
    data["var_3"].iloc[15] = np.nan

    with pytest.raises(ValueError):
        transformer = DropHighPSIFeatures(split_col="var_3")
        transformer.fit_transform(data)


def test_raise_error_missing_value_na(df):
    # Test an error is raised when missing values is set to raise
    data = df.copy()
    data["var_3"].iloc[15] = np.nan

    with pytest.raises(ValueError):
        transformer = DropHighPSIFeatures(missing_values="raise")
        transformer.fit(data)


def test_raise_missing_value_inf(df):
    # Test an error is raised when missing values is set to raise
    data = df.copy()
    data["var_3"].iloc[15] = np.inf

    with pytest.raises(ValueError):
        transformer = DropHighPSIFeatures(missing_values="raise")
        transformer.fit(data)


# ========= tests for _split_dataframe() fit ====


quantile_test = [(0.5, 50), (0.33, 33), (0.17, 17), (0.81, 81)]


@pytest.mark.parametrize("split_frac, expected", quantile_test)
def test_calculation_quantile(split_frac, expected):
    """Test the calculation of the quantiles using distinct values."""
    df = pd.DataFrame(
        {"A": [it for it in range(0, 101)], "B": [it for it in range(0, 101)]}
    )

    test = DropHighPSIFeatures(
        split_col="A", split_frac=split_frac, split_distinct=False
    )
    test.fit_transform(df)
    assert test.cut_off_ == expected


def test_calculation_distinct_value():
    """Test the calculation of the quantiles using distinct values."""
    df = pd.DataFrame(
        {"A": [it for it in range(0, 30)], "C": ["A", "B", "C", "D", "D", "D"] * 5}
    )

    test = DropHighPSIFeatures(split_col="C", split_frac=0.5, split_distinct=False)

    test.fit_transform(df)
    assert test.cut_off_ == "C"

    test = DropHighPSIFeatures(split_col="C", split_frac=0.5, split_distinct=True)
    test.fit_transform(df)
    assert test.cut_off_ == "B"


def test_calculation_df_split_with_different_types(df_mixed_types):
    """Test the split of the dataframe using different type of variables."""
    results = {}
    cut_offs = {}
    for split_col in df_mixed_types.columns:
        test = DropHighPSIFeatures(split_frac=0.5, split_col=split_col)
        test.fit_transform(df_mixed_types)
        results[split_col] = test.psi_values_
        cut_offs[split_col] = test.cut_off_

    assert results["A"] == pytest.approx({"B": 0.0}, 12)
    assert results["B"] == pytest.approx({"A": 3.0375978817052403}, 12)
    assert results["C"] == pytest.approx({"A": 2.27819841127893, "B": 0.0}, 12)
    assert results["time"] == pytest.approx({"A": 8.283089355027482, "B": 0.0}, 12)

    expected_cut_offs = {"A": 9.5, "B": 1.5, "C": "B", "time": date(2019, 1, 10)}

    assert cut_offs == expected_cut_offs

    # Test when no columns is defined
    test = DropHighPSIFeatures(split_frac=0.5)
    test.fit_transform(df_mixed_types)
    assert test.psi_values_ == pytest.approx({"A": 8.283089355027482, "B": 0.0}, 12)


def test_calculation_no_split_columns():
    """Test the split of the dataframe using different type of variables."""
    df = pd.DataFrame(
        {
            "time": [date(2012, 6, it) for it in range(1, 31)],
            "A": [it for it in range(0, 30)],
            "B": [1, 2, 3, 4, 5, 6] * 5,
        }
    )

    expected_psi = {"A": 7.745116608833365, "B": 0.1621860432432658}

    test = DropHighPSIFeatures(split_frac=0.5, split_distinct=True)
    test.fit_transform(df)

    assert test.cut_off_ == 14.5
    assert test.psi_values_ == pytest.approx(expected_psi)


type_test = [
    ("A", 14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    ("B", 1, [0, 3, 4, 7, 8, 11, 12, 15, 16, 19]),
    ("C", ["A"], [0, 4, 8, 12, 16]),
    ("time", date(2019, 1, 4), [0, 1, 2, 3]),
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
    # todo: can we compare dfs and not just shapes?
    """Test the cut off for different data types.

    For columns B, C and time we have 6 distinct values, 5 appearing 20 times and
    1 appearing 100 times. A 50% split based on the number of values will results
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


# TODO: are we testing _split_df() when user passes a list of values?
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


def test_variable_definition(df):
    """Test if defining the subset of features through the variable argument.

    Due to the low split fractions all variables will fail the PSI test. However
    the test shows that only those defined in the variable argument will are
    dropped from the dataframe.
    """
    #
    select = DropHighPSIFeatures(variables=["var_1", "var_3", "var_5"], split_frac=0.01)
    transformed_df = select.fit_transform(df)

    assert transformed_df.columns.to_list() == [
        "var_0",
        "var_2",
        "var_4",
        "drift_1",
        "drift_2",
    ]


# TODO: are we testing that the features are selected appropriately? i.e., the
# attribute features_to_drop_ ?

# TODO: can we add some tests for the transform method? comparing dfs?


def test_non_fitted_error(df):
    # when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer = DropHighPSIFeatures()
        transformer.transform(df)
