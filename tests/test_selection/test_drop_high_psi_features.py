from datetime import date

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from feature_engine.selection.drop_high_PSI_features import DropHighPSIFeatures


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


@pytest.fixture(scope="module")
def df_basis():
    # create array with 8 correlated features and 4 independent ones
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # transform array into pandas df
    colnames = ["var_" + str(i) for i in range(12)]
    X = pd.DataFrame(X, columns=colnames)

    return X


def test_sanity_checks(df_basis):
    """Sanity checks.

    All PSI must be zero if the two dataframe involved are the same.
    There will be no changes in the dataframe.
    """
    transformer = DropHighPSIFeatures(df_basis)
    X = transformer.fit_transform(df_basis)

    assert transformer.psi.value.min() == 0
    assert transformer.psi.value.max() == 0
    assert X.shape == df_basis.shape
    assert transformer.psi.shape[0] == df_basis.shape[1]


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
    df = pd.DataFrame({"A": [1, 1, 1, 4]})
    df2 = pd.DataFrame({"A": [4, 4, 4, 1]})

    ref_value = 1.098612

    test = DropHighPSIFeatures(
        df2,
        n_bins=2,
        method="equal_width",
        switch_basis=False,
        min_pct_empty_buckets=0.001,
    )

    test.fit(df)
    assert abs(test.psi.value[0] - ref_value) < 0.000001


def test_switch_basis():
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

    case = DropHighPSIFeatures(
        df_a, n_bins=5, switch_basis=False, min_pct_empty_buckets=0.001
    )
    case.fit(df_b)

    switch_case = DropHighPSIFeatures(
        df_a, n_bins=5, switch_basis=True, min_pct_empty_buckets=0.001
    )
    switch_case.fit(df_b)

    reverse_case = DropHighPSIFeatures(
        df_b, n_bins=5, switch_basis=True, min_pct_empty_buckets=0.001
    )
    reverse_case.fit(df_a)

    assert (case.psi == reverse_case.psi).all
    assert (case.psi != switch_case.psi).all


def test_split_df_according_to_time():

    df = pd.DataFrame(
        {
            "A": [it for it in range(0, 20)],
            "B": [1, 2, 3, 4] * 5,
            "time": [date(2019, 1, it + 1) for it in range(20)],
        }
    )

    below_value = df[df["time"] <= date(2019, 1, 11)]
    above_value = df[df["time"] > date(2019, 1, 11)]

    sb = False
    basis = {"date_col": "time", "cut_off_date": date(2019, 1, 11)}
    cut_off = DropHighPSIFeatures(
        basis, n_bins=5, switch_basis=sb, min_pct_empty_buckets=0.001
    )
    cut_off.fit(df).psi

    separate_df = DropHighPSIFeatures(
        below_value, n_bins=5, switch_basis=sb, min_pct_empty_buckets=0.001
    )
    separate_df.fit(above_value).psi

    assert (cut_off.fit(df).psi == separate_df.fit(above_value).psi).all


def test_value_error_is_raised(df):

    with pytest.raises(ValueError):
        basis = pd.Series([1, 2, 3])
        DropHighPSIFeatures(basis, n_bins=5, min_pct_empty_buckets=0.001)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(df, n_bins=1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(df, threshold=-1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(df, switch_basis=1)

    with pytest.raises(ValueError):
        DropHighPSIFeatures(df, method="unknown")


def test_variable_definition(df):

    df_0 = df * 0
    select = DropHighPSIFeatures(df, variables=["var_1", "var_3", "var_5"])
    transformed_df = select.fit_transform(df_0)

    assert transformed_df.columns.to_list() == ["var_0", "var_2", "var_4"]


def test_non_fitted_error(df):
    # when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer = DropHighPSIFeatures(df)
        transformer.transform(df)
