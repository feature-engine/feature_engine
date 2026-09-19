import math
import re
from datetime import date, datetime

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from feature_engine.selection import DropHighPSIFeatures
from tests.backend_helpers import frame_to_dict

# The expected PSI values were determined with the Probatus package:
# AutoDist(statistical_tests=["PSI"], binning_strategies="QuantileBucketer",
# bin_count=10).compute(data.iloc[0:500, :], data.iloc[500:, :])
EXPECTED_PSI = {
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

NUMERICAL = ["var_0", "var_1", "var_2", "var_3", "var_4", "var_5", "drift_1", "drift_2"]


@pytest.fixture(scope="module")
def data_drift():
    """6 features without drift, 2 numerical and 1 categorical feature that drift
    between the first and the second half of the rows."""
    X, _ = make_classification(
        n_samples=1000,
        n_features=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )
    data = {f"var_{i}": X[:, i].tolist() for i in range(6)}
    data["cat_1"] = ["A", "B"] * 500
    data["drift_1"] = list(range(1000))
    data["drift_2"] = [number / 2 for number in range(1000)]
    data["drift_cat_1"] = ["A"] * 500 + ["B"] * 500
    return data


DATA_MIXED = {
    "A": list(range(20)),
    "B": [1, 2, 2, 1] * 5,
    "C": ["A", "B", "D", "D"] * 5,
    "time": [datetime(2019, 1, day + 1) for day in range(20)],
}


def basis_rows(transformer, X):
    """Positions of the rows that go to the basis dataset."""
    is_basis = transformer._basis_mask(X, nw.from_native(X, eager_only=True))
    return np.flatnonzero(is_basis).tolist()


# init parameters
@pytest.mark.parametrize("split_col", [["hola"], 1.5, {"a": 1}])
def test_error_if_split_col_not_allowed(split_col):
    msg = f"split_col must be a string an integer or None. Got {split_col} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(split_col=split_col)


def test_error_if_split_frac_and_cut_off_are_none():
    msg = (
        "cut_off and split_frac cannot be both set to None. The current values are "
        "(None, None). Please specify a value for at least one of these parameters."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(split_frac=None, cut_off=None)


@pytest.mark.parametrize("split_frac", [0, 1, -0.5, 1.5, "0.5", [0.5]])
def test_error_if_split_frac_not_allowed(split_frac):
    msg = f"split_frac must be a float between 0 and 1. Got {split_frac} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(split_frac=split_frac)


@pytest.mark.parametrize("split_distinct", [1, "True", None, [True]])
def test_error_if_split_distinct_not_bool(split_distinct):
    msg = f"split_distinct must be a boolean. Got {split_distinct} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(split_distinct=split_distinct)


@pytest.mark.parametrize("switch", [1, "True", None, [True]])
def test_error_if_switch_not_bool(switch):
    msg = f"switch must be a boolean. Got {switch} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(switch=switch)


@pytest.mark.parametrize("threshold", [-1, -0.1, "hola", None, [0.1]])
def test_error_if_threshold_not_allowed(threshold):
    msg = f"threshold must be greater than 0 or 'auto'. Got {threshold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(threshold=threshold)


@pytest.mark.parametrize("bins", [1, 0, -3, 2.5, "10", None])
def test_error_if_bins_not_allowed(bins):
    msg = f"bins must be an integer >= 2. Got {bins} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(bins=bins)


@pytest.mark.parametrize("strategy", ["unknown", ["equal_width"], None, 1])
def test_error_if_strategy_not_allowed(strategy):
    msg = (
        "strategy takes only values equal_width or equal_frequency. "
        f"Got {strategy} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(strategy=strategy)


@pytest.mark.parametrize("min_pct_empty_bins", [-1, -0.1, "unknown", None, [0.1]])
def test_error_if_min_pct_empty_bins_not_allowed(min_pct_empty_bins):
    msg = f"min_pct_empty_bins must be >= 0. Got {min_pct_empty_bins} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(min_pct_empty_bins=min_pct_empty_bins)


@pytest.mark.parametrize("missing_values", ["hola", ["raise"], None, 1])
def test_error_if_missing_values_not_allowed(missing_values):
    msg = (
        "missing_values takes only values 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(missing_values=missing_values)


def test_error_if_split_col_in_variables():
    msg = (
        "hola cannot be used to split the data and be evaluated at the same time. "
        "Either remove hola from the variables list or choose another splitting "
        "criteria."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(split_col="hola", variables=["hola", "chau"])


@pytest.mark.parametrize("p_value", ["hola", -1.0, 10.0, 1, None])
def test_error_if_p_value_not_allowed(p_value):
    msg = f"p_value must be a float between 0 and 1. Got {p_value} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(p_value=p_value)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {
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
            "confirm_variables": True,
            "p_value": 0.2,
        },
        {"split_col": 0, "split_frac": None, "cut_off": 0, "threshold": "auto"},
        {"cut_off": date(2019, 1, 1), "threshold": 1, "min_pct_empty_bins": 0},
    ],
)
def test_init_param_assignment(params):
    defaults = {
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
        "confirm_variables": False,
        "p_value": 0.001,
    }
    transformer = DropHighPSIFeatures(**params)
    for param, value in {**defaults, **params}.items():
        assert getattr(transformer, param) == value


# fit and transform
@pytest.mark.parametrize(
    "variables, expected_variables",
    [
        (None, NUMERICAL),
        ("all", NUMERICAL + ["cat_1", "drift_cat_1"]),
        (
            ["var_2", "var_3", "drift_1", "drift_2"],
            ["var_2", "var_3", "drift_1", "drift_2"],
        ),
        (["cat_1", "drift_cat_1"], ["cat_1", "drift_cat_1"]),
        (["var_0", "drift_cat_1", "drift_1"], ["var_0", "drift_1", "drift_cat_1"]),
        ("var_0", ["var_0"]),
    ],
)
@pytest.mark.parametrize("threshold", [0.25, "auto"])
def test_fit_attributes_and_transform(
    make_df, data_drift, variables, expected_variables, threshold
):
    X = make_df(data_drift)
    transformer = DropHighPSIFeatures(variables=variables, threshold=threshold)
    Xt = transformer.fit_transform(X)

    features_to_drop = [var for var in expected_variables if "drift" in var]
    assert transformer.variables_ == expected_variables
    assert transformer.psi_values_ == pytest.approx(
        {var: EXPECTED_PSI[var] for var in expected_variables}
    )
    assert transformer.features_to_drop_ == features_to_drop
    assert transformer.cut_off_ == 499.5
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        var: values
        for var, values in data_drift.items()
        if var not in features_to_drop
    }


def test_auto_threshold_calculation():
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


def test_auto_threshold_drops_features(make_df, data_drift):
    # with p_value=0.2 the threshold for 10 intervals and 500 + 500 rows is 0.049.
    transformer = DropHighPSIFeatures(threshold="auto", p_value=0.2, variables="all")
    transformer.fit(make_df(data_drift))
    assert transformer.features_to_drop_ == [
        "var_4",
        "drift_1",
        "drift_2",
        "drift_cat_1",
    ]


def test_psi_with_equal_width_strategy(make_df, data_drift):
    transformer = DropHighPSIFeatures(strategy="equal_width")
    transformer.fit(make_df(data_drift))

    assert transformer.psi_values_ == pytest.approx(
        {
            "var_0": 0.014858665472468786,
            "var_1": 0.04514737836588022,
            "var_2": 0.03431479397506742,
            "var_3": 0.04298209189840294,
            "var_4": 0.02385796430263416,
            "var_5": 0.046809664317794444,
            "drift_1": 8.283089355027482,
            "drift_2": 8.283089355027482,
        },
    )
    assert transformer.features_to_drop_ == ["drift_1", "drift_2"]


def test_empty_bins_take_min_pct_empty_bins(make_df):
    X = make_df({"x": [1, 2, 3, 4], "cat": ["A", "B", "A", "C"]})
    transformer = DropHighPSIFeatures(
        variables="all", bins=2, min_pct_empty_bins=0.01
    ).fit(X)

    # "B" is only in the basis set and "C" only in the test set.
    psi_cat = (0.01 - 0.5) * math.log(0.01 / 0.5) + (0.5 - 0.01) * math.log(0.5 / 0.01)
    # the basis set, [1, 2], has one value per interval; the test set, [3, 4], has
    # both values in the second interval.
    psi_x = (0.01 - 0.5) * math.log(0.01 / 0.5) + (1 - 0.5) * math.log(1 / 0.5)
    assert transformer.psi_values_ == pytest.approx({"x": psi_x, "cat": psi_cat})


def test_constant_feature_has_psi_zero(make_df):
    X = make_df({"x": [1.5] * 20, "y": list(range(20))})
    transformer = DropHighPSIFeatures(bins=3).fit(X)
    assert transformer.psi_values_["x"] == 0


def test_split_col_not_included_in_variables(make_df, data_drift):
    X = make_df(data_drift)

    transformer = DropHighPSIFeatures(split_col="var_3", variables=None).fit(X)
    assert transformer.variables_ == [var for var in NUMERICAL if var != "var_3"]

    transformer = DropHighPSIFeatures(split_col="var_3", variables="all").fit(X)
    assert "var_3" not in transformer.variables_
    assert "var_3" not in transformer.psi_values_

    transformer = DropHighPSIFeatures(split_col="cat_1", variables="all").fit(X)
    assert transformer.variables_ == NUMERICAL + ["drift_cat_1"]
    assert "cat_1" not in transformer.psi_values_


def test_error_if_split_col_not_in_df(make_df, data_drift):
    X = make_df({var: data_drift[var] for var in ["var_1", "var_2"]})
    transformer = DropHighPSIFeatures(split_col="var_0")
    with pytest.raises(ValueError, match=re.escape("var_0 is not in the dataframe.")):
        transformer.fit(X)


@pytest.mark.parametrize(
    "variables, missing, expected_variables",
    [
        (["var_2", "var_3", "drift_1", "drift_2"], "drift_1", ["var_2", "var_3"]),
        (["var_0", "drift_1", "drift_cat_1"], "drift_cat_1", ["var_0", "drift_1"]),
    ],
)
def test_confirm_variables(make_df, data_drift, variables, missing, expected_variables):
    X = make_df({var: values for var, values in data_drift.items() if var != missing})
    transformer = DropHighPSIFeatures(variables=variables, confirm_variables=True)
    transformer.fit(X)

    assert transformer.variables_ == expected_variables + (
        ["drift_2"] if "drift_2" in variables else []
    )
    assert transformer.psi_values_ == pytest.approx(
        {var: EXPECTED_PSI[var] for var in transformer.variables_}
    )


def test_error_if_no_numerical_variables(make_df, data_drift):
    X = make_df({var: data_drift[var] for var in ["cat_1", "drift_cat_1"]})
    msg = (
        "No numerical variables found in this dataframe. Check variable dtypes or "
        "set return_empty to True to return an empty list instead."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        DropHighPSIFeatures().fit(X)


def test_error_if_confirm_variables_returns_empty_list(make_df, data_drift):
    X = make_df({var: data_drift[var] for var in NUMERICAL})
    transformer = DropHighPSIFeatures(
        variables=["cat_1", "drift_cat_1"], confirm_variables=True
    )
    msg = "None of the variables in the list are present in the dataframe."
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(X)


@pytest.mark.parametrize("split_col", ["var_3", "cat_1"])
def test_error_if_na_in_split_col(make_df, data_drift, split_col):
    data = {var: list(values) for var, values in data_drift.items()}
    data[split_col][15] = None
    data[split_col][17] = None
    transformer = DropHighPSIFeatures(split_col=split_col, missing_values="ignore")
    msg = (
        "There are 2 missing values in the reference variable. Missing data are not "
        "allowed in the variable used to split the dataframe."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(make_df(data))


@pytest.mark.parametrize("variable", ["var_3", "cat_1"])
def test_error_if_na_in_variables(make_df, data_drift, variable):
    data = {var: list(values) for var, values in data_drift.items()}
    data[variable][15] = None
    transformer = DropHighPSIFeatures(variables="all", missing_values="raise")
    msg = (
        "Some of the variables in the dataset contain NaN. Check and remove those "
        "before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(make_df(data))


def test_missing_values_ignored(make_df, data_drift):
    data = {var: list(values) for var, values in data_drift.items()}
    data["var_3"][15] = None
    data["cat_1"][15] = None
    X = make_df(data)
    transformer = DropHighPSIFeatures(variables="all", missing_values="ignore")
    Xt = transformer.fit_transform(X)

    # a missing value in a feature does not remove the observation from the others
    expected_psi = dict(
        EXPECTED_PSI, var_3=0.03404846717863001, cat_1=4.016053504547722e-06
    )
    assert transformer.psi_values_ == pytest.approx(expected_psi)
    assert transformer.features_to_drop_ == ["drift_1", "drift_2", "drift_cat_1"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        var: values for var, values in data.items() if "drift" not in var
    }


@pytest.mark.parametrize("missing_values", ["raise", "ignore"])
def test_error_if_inf_in_variables(make_df, data_drift, missing_values):
    data = {var: list(values) for var, values in data_drift.items()}
    data["var_3"][15] = np.inf
    transformer = DropHighPSIFeatures(missing_values=missing_values)
    msg = (
        "Some of the variables to transform contain inf values. Check and remove "
        "those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(make_df(data))


@pytest.mark.parametrize(
    "rows, n_basis, n_test", [(range(500), 0, 500), (range(500, 1000), 500, 0)]
)
def test_error_if_only_missing_values_in_basis_or_test(
    make_df, data_drift, rows, n_basis, n_test
):
    data = {var: data_drift[var] for var in ["var_0", "var_1"]}
    data["var_1"] = [None if row in rows else v for row, v in enumerate(data["var_1"])]
    transformer = DropHighPSIFeatures(missing_values="ignore")
    msg = (
        "The variable var_1 has only missing values in the basis or in the test set, "
        f"so its PSI can't be computed. Got {n_basis} values in the basis set and "
        f"{n_test} values in the test set."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(make_df(data))


@pytest.mark.parametrize(
    "cut_off, n_basis, n_test", [(5, 6, 994), (-1, 0, 1000), (5000, 1000, 0)]
)
def test_error_if_too_few_rows_in_basis_or_test(
    make_df, data_drift, cut_off, n_basis, n_test
):
    X = make_df({var: data_drift[var] for var in NUMERICAL})
    msg = (
        "The number of rows in the basis and test datasets that will be used in the "
        "PSI calculations must be at least larger than 10. After splitting the "
        "original dataset based on the given cut_off or split_frac we have "
        f"{n_basis} samples in the basis set, and {n_test} samples in the test set. "
        "Please adjust the value of the cut_off or split_frac."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropHighPSIFeatures(cut_off=cut_off).fit(X)


@pytest.mark.parametrize(
    "split_frac, expected", [(0.5, 50), (0.33, 33), (0.17, 17), (0.81, 81)]
)
def test_cut_off_from_split_frac(make_df, split_frac, expected):
    X = make_df({"A": list(range(101)), "B": list(range(101))})
    transformer = DropHighPSIFeatures(split_col="A", split_frac=split_frac)
    transformer.fit(X)
    assert transformer.cut_off_ == expected


@pytest.mark.parametrize("value, fraction", [(50, 50), (1, 30), (10, 40), (7, 80)])
def test_cut_off_from_split_frac_with_skewed_variable(make_df, value, fraction):
    X = make_df(
        {
            "A": [value] * (fraction + 1) + list(range(fraction + 1, 101)),
            "B": list(range(101)),
        }
    )
    transformer = DropHighPSIFeatures(split_col="A", split_frac=fraction / 100)
    transformer.fit(X)
    assert transformer.cut_off_ == value


@pytest.mark.parametrize(
    "values, split_distinct, expected",
    [
        (["A", "B", "C", "D", "D", "D"], False, "C"),
        (["A", "B", "C", "D", "D", "D"], True, "B"),
        (["A", "A", "A", "B", "C", "D"], False, "A"),
        (["A", "A", "A", "B", "C", "D"], True, "B"),
    ],
)
def test_cut_off_from_split_frac_with_categorical_variable(
    make_df, values, split_distinct, expected
):
    X = make_df({"A": list(range(30)), "C": values * 5})
    transformer = DropHighPSIFeatures(split_col="C", split_distinct=split_distinct)
    transformer.fit(X)
    assert transformer.cut_off_ == expected


def test_psi_and_cut_off_with_different_split_col_types(make_df):
    X = make_df(DATA_MIXED)
    expected = {
        "A": ({"B": 0.0, "C": 0.1621860432432657}, 9.5),
        "B": ({"A": 3.0375978817052403, "C": 8.515489752777954}, 1.5),
        "C": ({"A": 2.27819841127893, "B": 0.0}, "B"),
        "time": (
            {"A": 8.283089355027482, "B": 0.0, "C": 0.1621860432432657},
            datetime(2019, 1, 10),
        ),
        None: ({"A": 8.283089355027482, "B": 0.0, "C": 0.1621860432432657}, 9.5),
    }
    for split_col, (psi_values, cut_off) in expected.items():
        transformer = DropHighPSIFeatures(split_col=split_col, variables="all")
        transformer.fit(X)
        assert transformer.psi_values_ == pytest.approx(psi_values)
        assert transformer.cut_off_ == cut_off


@pytest.mark.parametrize(
    "split_distinct, expected",
    [(True, [0, 1, 2, 3, 4, 7, 8]), (False, [0, 1, 4, 7, 8])],
)
def test_split_distinct_with_numerical_values(make_df, split_distinct, expected):
    X = make_df(
        {
            "ID": [1, 1, 2, 3, 1, 4, 5, 1, 1, 6],
            "numerical": [1, 1, 1, 4, 1, 4, 3, 7, 1, 3],
        }
    )
    transformer = DropHighPSIFeatures(split_col="ID", split_distinct=split_distinct)
    assert basis_rows(transformer, X) == expected


@pytest.mark.parametrize(
    "split_col, cut_off, expected",
    [
        ("A", 14, list(range(15))),
        ("B", 1, [0, 3, 4, 7, 8, 11, 12, 15, 16, 19]),
        ("C", ["B"], [1, 5, 9, 13, 17]),
        ("C", "B", [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]),
        ("time", datetime(2019, 1, 4), [0, 1, 2, 3]),
        ("A", [1, 2, 10, 11, 16], [1, 2, 10, 11, 16]),
        ("B", [2], [1, 2, 5, 6, 9, 10, 13, 14, 17, 18]),
        ("C", ["B", "D"], [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]),
        ("time", [date(2019, 1, day) for day in [1, 2, 5, 18]], [0, 1, 4, 17]),
        ("time", [datetime(2019, 1, day) for day in [1, 2, 5, 18]], [0, 1, 4, 17]),
        (None, 5, [0, 1, 2, 3, 4, 5]),
        (None, [3, 7, 11], [3, 7, 11]),
    ],
)
def test_split_with_cut_off(make_df, split_col, cut_off, expected):
    X = make_df(DATA_MIXED)
    transformer = DropHighPSIFeatures(split_col=split_col, cut_off=cut_off)
    assert basis_rows(transformer, X) == expected


def test_cut_off_zero_is_used(make_df):
    X = make_df({"A": [0, 1, 2, 3] * 10, "B": list(range(40))})
    transformer = DropHighPSIFeatures(split_col="A", cut_off=0, bins=5).fit(X)
    assert transformer.cut_off_ == 0
    assert basis_rows(transformer, X) == list(range(0, 40, 4))


@pytest.mark.parametrize("split_col", ["A", "B", "C", "time"])
def test_split_distinct(make_df, split_col):
    # 6 distinct values, 5 appear 20 times and 1 appears 100 times: the basis gets
    # the rows of the first 3 distinct values.
    X = make_df(
        {
            "A": list(range(200)),
            "B": [1, 2, 3, 4, 5, 6, 6, 6, 6, 6] * 20,
            "C": ["A", "B", "C", "D", "E", "F", "F", "F", "F", "F"] * 20,
            "time": [date(2019, 1, day + 1) for day in range(5)] * 20
            + [date(2019, 1, 31)] * 100,
        }
    )
    if split_col == "A":
        expected = list(range(100))
    elif split_col == "time":
        expected = [row for row in range(100) if row % 5 < 3]
    else:
        expected = [row for row in range(200) if row % 10 < 3]
    transformer = DropHighPSIFeatures(split_col=split_col, split_distinct=True)
    assert basis_rows(transformer, X) == expected


@pytest.mark.parametrize(
    "split_frac, cut_off, expected_cut_off, n_basis",
    [(0.5, None, 499.5, 500), (0.6, None, 599.4, 600), (0.5, 250, 250, 251)],
)
def test_split_by_row_order_when_split_col_is_none(
    make_df, data_drift, split_frac, cut_off, expected_cut_off, n_basis
):
    # pandas uses the index, here the default index is the position of the rows.
    X = make_df(data_drift)
    transformer = DropHighPSIFeatures(split_frac=split_frac, cut_off=cut_off).fit(X)
    assert transformer.cut_off_ == pytest.approx(expected_cut_off)
    assert basis_rows(transformer, X) == list(range(n_basis))


@pytest.mark.parametrize(
    "split_frac, cut_off, expected_cut_off, expected",
    [
        (0.5, None, 499.5, set(range(500))),
        (0.6, None, 599.4, set(range(600))),
        (0.5, 250, 250, set(range(251))),
    ],
)
def test_split_by_index_of_shuffled_pandas_df(
    data_drift, split_frac, cut_off, expected_cut_off, expected
):
    X = pd.DataFrame(data_drift).sample(frac=1, random_state=0)
    transformer = DropHighPSIFeatures(split_frac=split_frac, cut_off=cut_off)
    is_basis = transformer._basis_mask(X, nw.from_native(X))
    assert transformer.cut_off_ == pytest.approx(expected_cut_off)
    assert set(X.index[is_basis]) == expected


def test_split_by_datetime_index_of_pandas_df(data_drift):
    X = pd.DataFrame(data_drift)
    X.index = pd.date_range("2020-01-01", periods=1000, freq="h")
    transformer = DropHighPSIFeatures(cut_off=pd.Timestamp("2020-01-20")).fit(X)
    assert transformer.psi_values_ == pytest.approx(
        {
            "var_0": 0.026166596537012143,
            "var_1": 0.05683862097207795,
            "var_2": 0.04828999318701926,
            "var_3": 0.03704211767222722,
            "var_4": 0.07053922062598025,
            "var_5": 0.011565170354399454,
            "drift_1": 8.270551209760598,
            "drift_2": 8.270551209760598,
        },
    )


def test_switch(make_df):
    data_a = {
        "a": [1.0, 2, 3, 1],
        "b": [1.0, 2, 3, 4],
        "c": [1, 2, 3, 4],
        "d": [1.7, 4.7, 6.6, 7.8],
    }
    data_b = {
        "a": [4.0, 3, 5, 1],
        "b": [11.0, 1, 2, 4],
        "c": [4, 2, 2, 4],
        "d": [4.7, 4.7, 7.6, 7.8],
    }
    X_order = make_df({var: data_a[var] + data_b[var] for var in data_a})
    X_reverse = make_df({var: data_b[var] + data_a[var] for var in data_a})

    case = DropHighPSIFeatures(bins=3, switch=False, min_pct_empty_bins=0.001)
    case.fit(X_order)
    switch_case = DropHighPSIFeatures(bins=3, switch=True, min_pct_empty_bins=0.001)
    switch_case.fit(X_reverse)

    assert case.psi_values_ == switch_case.psi_values_
    assert case.psi_values_ == pytest.approx(
        {
            "a": 1.0986122886681098,
            "b": 1.5481305636876856,
            "c": 1.5481305636876856,
            "d": 1.5481305636876856,
        }
    )


def test_transform_restores_train_column_order(make_df, data_drift):
    X = make_df(data_drift)
    transformer = DropHighPSIFeatures().fit(X)
    X_reordered = make_df({var: data_drift[var] for var in reversed(data_drift)})
    Xt = transformer.transform(X_reordered)

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == [
        var for var in data_drift if var not in ["drift_1", "drift_2"]
    ]


def test_error_if_transform_df_has_different_number_of_columns(make_df, data_drift):
    transformer = DropHighPSIFeatures().fit(make_df(data_drift))
    X = make_df({**data_drift, "A": [1] * 1000})
    msg = (
        "The number of columns in this dataset is different from the one used to "
        "fit this transformer (when using the fit() method)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.transform(X)


def test_error_if_not_fitted(make_df, data_drift):
    msg = (
        "This DropHighPSIFeatures instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        DropHighPSIFeatures().transform(make_df(data_drift))


def test_input_df_is_not_modified(make_df, data_drift):
    X = make_df(data_drift)
    DropHighPSIFeatures(variables="all", switch=True).fit_transform(X)
    assert frame_to_dict(X) == data_drift


def test_category_dtype(data_drift):
    X = pd.DataFrame(data_drift)
    X["drift_cat_1"] = X["drift_cat_1"].astype("category")
    X["cat_1"] = pd.Categorical(X["cat_1"], categories=["C", "B", "A"])
    transformer = DropHighPSIFeatures(variables=["cat_1", "drift_cat_1"])
    Xt = transformer.fit_transform(X)

    assert transformer.psi_values_ == pytest.approx(
        {"cat_1": 0.0, "drift_cat_1": 18.41883867587797}
    )
    pd.testing.assert_frame_equal(Xt, X.drop(columns="drift_cat_1"))


def test_integer_column_names(data_drift):
    X = pd.DataFrame(data_drift)
    X.columns = list(range(X.shape[1]))
    transformer = DropHighPSIFeatures(split_col=6, variables="all")
    Xt = transformer.fit_transform(X)

    # the column cat_1, which alternates A and B, puts even rows in the basis set.
    assert transformer.cut_off_ == "A"
    assert transformer.variables_ == [0, 1, 2, 3, 4, 5, 7, 8, 9]
    assert transformer.features_to_drop_ == []
    pd.testing.assert_frame_equal(Xt, X)

    transformer = DropHighPSIFeatures(variables=[0, 7, 9]).fit(X)
    assert transformer.psi_values_ == pytest.approx(
        {
            0: EXPECTED_PSI["var_0"],
            7: EXPECTED_PSI["drift_1"],
            9: EXPECTED_PSI["drift_cat_1"],
        },
    )
    pd.testing.assert_frame_equal(transformer.transform(X), X.drop(columns=[7, 9]))
