from datetime import date

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from feature_engine.selection.drop_psi_features import DropHighPSIFeatures


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

    return X


def test_sanity_checks(df):
    """Sanity checks.

    All PSI must be zero if the two dataframe involved are the same.
    There will be no changes in the dataframe.
    """
    transformer = DropHighPSIFeatures()
    X = transformer.fit_transform(df)

    assert min(transformer.psi_values_.values()) >= 0
    assert X.shape == df.shape
    assert len(transformer.psi_values_.values()) == df.shape[1]


def test_check_psi_values():
    """Compare the PSI value with a reference.

    As reference we take the implementation from the Probatus package.
    ```
    from probatus.stat_tests import AutoDist

    df = pd.DataFrame({"A": [1, 1, 1, 4]})
    df2 = pd.DataFrame({"A": [4, 4, 4, 1]})

    (AutoDist(statistical_tests=['PSI'],
    binning_strategies="SimpleBucketer",
    bin_count=2)
    .compute(df, df2)
    )
    ```
    Output; 1.098612
    """
    df = pd.DataFrame({"A": [1, 1, 1, 4, 4, 4, 4, 1], "B": [1, 1, 1, 1, 2, 2, 2, 2]})

    ref_value = 1.098612

    test = DropHighPSIFeatures(
        split_frac=0.5,
        split_col="B",
        bins=2,
        strategy="equal_width",
        switch=False,
        min_pct_empty_buckets=0.001,
    )

    test.fit(df)

    assert abs(test.psi_values_["A"] - ref_value) < 0.000001


quantile_test = [(0.5, 50), (0.33, 33), (0.17, 17), (0.81, 81)]


@pytest.mark.parametrize("split_frac, expected", quantile_test)
def test_calculation_quantile(split_frac, expected):
    """Test the calculation of the quantiles using distinct values."""
    df = pd.DataFrame(
        {"A": [it for it in range(1, 101)], "B": [it for it in range(1, 101)]}
    )

    test = DropHighPSIFeatures(
        split_col="A", split_frac=split_frac, split_distinct_value=False
    )
    test.fit_transform(df)
    assert test.cut_off == expected


def test_calculation_distinct_value():
    """Test the calculation of the quantiles using distinct values."""
    df = pd.DataFrame(
        {"A": [it for it in range(0, 30)], "C": ["A", "B", "C", "D", "D", "D"] * 5}
    )

    test = DropHighPSIFeatures(
        split_col="C", split_frac=0.5, split_distinct_value=False
    )
    test.fit_transform(df)
    assert test.cut_off == "C"

    test = DropHighPSIFeatures(split_col="C", split_frac=0.5, split_distinct_value=True)
    test.fit_transform(df)
    assert test.cut_off == "B"


@pytest.fixture(params=["time", "A", "B", "C"])
def test_column(request):
    return request.param


def test_calculation_df_split_with_different_types(test_column):
    """Test the split of the dataframe using different type of variables."""
    df = pd.DataFrame(
        {
            "time": [date(2012, 6, it) for it in range(1, 31)],
            "A": [it for it in range(0, 30)],
            "B": [1, 2, 3, 4, 5, 6] * 5,
            "C": ["A", "B", "C", "D", "D", "D"] * 5,
        }
    )

    test = DropHighPSIFeatures(
        split_col=test_column, split_frac=0.5, split_distinct_value=False
    )
    results = test.fit_transform(df)
    assert results.shape[0] > 0
    assert results.shape[1] > 0


def test_calculation_no_split_columns():
    """Test the split of the dataframe using different type of variables."""
    df = pd.DataFrame(
        {
            "time": [date(2012, 6, it) for it in range(1, 31)],
            "A": [it for it in range(0, 30)],
            "B": [1, 2, 3, 4, 5, 6] * 5,
        }
    )

    test = DropHighPSIFeatures(split_frac=0.5, split_distinct_value=True)
    test.fit_transform(df)
    assert len(test.psi_values_) == 2


def test_switch():
    """Test the functionality to switch the basis."""
    import pandas as pd

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
        split_frac=0.5, bins=5, switch=False, min_pct_empty_buckets=0.001
    )
    case.fit(df_order)

    switch_case = DropHighPSIFeatures(
        split_frac=0.5, bins=5, switch=True, min_pct_empty_buckets=0.001
    )
    switch_case.fit(df_reverse)

    assert case.psi_values_ == switch_case.psi_values_


def test_split_df_according_to_col():

    df = pd.DataFrame(
        {
            "A": [it for it in range(0, 20)],
            "B": [1, 2, 3, 4] * 5,
            "time": [date(2019, 1, it + 1) for it in range(20)],
        }
    )

    cut_off = DropHighPSIFeatures(
        split_col="time", split_frac=0.5, bins=5, min_pct_empty_buckets=0.001
    )
    psi = cut_off.fit(df).psi_values_

    assert len(psi) == 2


def test_value_error_is_raised(df):

    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_frac=0, bins=5, min_pct_empty_buckets=0.001)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(split_frac=1, bins=5, min_pct_empty_buckets=0.001)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(bins=1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(threshold=-1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(switch=1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(strategy="unknown")


def test_variable_definition(df):

    select = DropHighPSIFeatures(variables=["var_1", "var_3", "var_5"], split_frac=0.01)
    transformed_df = select.fit_transform(df)

    assert transformed_df.columns.to_list() == ["var_0", "var_2", "var_4"]


def test_non_fitted_error(df):
    # when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer = DropHighPSIFeatures()
        transformer.transform(df)
