import re

import pandas as pd
import pytest

from feature_engine.imputation import MeanImputer, MeanMedianImputer

DEPRECATION_WARNING = (
    "MeanMedianImputer was deprecated in favour of MeanImputer in version "
    "2.0.0 and will be removed in version 2.1.0. To silence this warning, "
    "use MeanImputer instead."
)


@pytest.fixture(
    params=[MeanImputer, MeanMedianImputer],
    ids=["MeanImputer", "MeanMedianImputer"],
)
def imputer_class(request):
    return request.param


def make_imputer(imputer_class, **kwargs):
    if imputer_class is MeanMedianImputer:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            return imputer_class(**kwargs)
    return imputer_class(**kwargs)


def test_mean_median_imputer_raises_future_warning():
    with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
        MeanMedianImputer()


def test_mean_imputation_and_automatically_select_variables(df_na, imputer_class):
    # set up transformer
    imputer = make_imputer(imputer_class, imputation_method="mean", variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # set up reference result
    X_reference = df_na.copy()
    X_reference["Age"] = X_reference["Age"].fillna(28.714285714285715)
    X_reference["Marks"] = X_reference["Marks"].fillna(0.6833333333333332)

    # test init params
    assert imputer.imputation_method == "mean"
    assert imputer.variables is None

    # test fit attributes
    assert imputer.variables_ == ["Age", "Marks"]
    imputer.imputer_dict_ = {
        key: round(value, 3) for (key, value) in imputer.imputer_dict_.items()
    }
    assert imputer.imputer_dict_ == {
        "Age": 28.714,
        "Marks": 0.683,
    }
    assert imputer.n_features_in_ == 6

    # test transform output:
    # selected variables should have no NA
    # not selected variables should still have NA
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() == 0
    assert X_transformed[["Name", "City"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_median_imputation_when_user_enters_single_variables(df_na, imputer_class):
    # set up trasnformer
    imputer = make_imputer(imputer_class, imputation_method="median", variables=["Age"])
    X_transformed = imputer.fit_transform(df_na)

    # set up reference output
    X_reference = df_na.copy()
    X_reference["Age"] = X_reference["Age"].fillna(23.0)

    # test init params
    assert imputer.imputation_method == "median"
    assert imputer.variables == ["Age"]

    # test fit attributes
    assert imputer.n_features_in_ == 6
    assert imputer.imputer_dict_ == {"Age": 23.0}

    # test transform output
    assert X_transformed["Age"].isnull().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_error_with_wrong_imputation_method(imputer_class):
    with pytest.raises(ValueError):
        make_imputer(imputer_class, imputation_method="arbitrary")


@pytest.mark.parametrize(
    ("method", "expected_value"),
    [("mean", 4.0), ("median", 3.0)],
)
def test_mean_imputer_fit_reads_input_and_transform_remains_independent(
    method, expected_value
):
    frame = pd.DataFrame(
        {"value": [1.0, 3.0, 8.0, float("nan")], "label": ["a", "b", "c", "d"]},
        index=pd.Index([7, 3, 9, 1], name="row"),
    )
    original = frame.copy(deep=True)
    imputer = MeanImputer(imputation_method=method, variables=["value"])

    assert imputer.fit(frame) is imputer
    pd.testing.assert_frame_equal(frame, original)
    assert imputer.imputer_dict_ == {"value": expected_value}
    assert imputer.variables_ == ["value"]
    assert imputer.feature_names_in_ == list(frame.columns)

    expected = original.copy(deep=True)
    expected.loc[1, "value"] = expected_value
    result = imputer.transform(frame)
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(frame, original)
    result.loc[7, "value"] = 100.0
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("method", ["mean", "median"])
def test_mean_imputer_numpy_fit_does_not_mutate_input(method):
    array = pd.DataFrame({"x0": [1.0, 3.0], "x1": [float("nan"), 4.0]}).to_numpy()
    original = array.copy()
    imputer = MeanImputer(imputation_method=method)

    result = imputer.fit_transform(array)
    assert list(result.columns) == ["x0", "x1"]
    assert result.loc[0, "x1"] == 4.0
    assert imputer.imputer_dict_["x1"] == 4.0
    pd.testing.assert_frame_equal(pd.DataFrame(array), pd.DataFrame(original))


@pytest.mark.parametrize("method", ["mean", "median"])
def test_mean_imputer_nullable_extensions_and_reordered_transform(method):
    train = pd.DataFrame(
        {"nullable_float": [1.0, pd.NA, 3.0], "nullable_int": [1, 3, 5],
         "label": ["a", "b", "c"]},
        index=pd.Index([8, 2, 5], name="row"),
    ).astype({"nullable_float": "Float64", "nullable_int": "Int64"})
    original = train.copy(deep=True)
    imputer = MeanImputer(
        imputation_method=method, variables=["nullable_float", "nullable_int"]
    )
    imputer.fit(train)
    pd.testing.assert_frame_equal(train, original)
    assert imputer.imputer_dict_ == {"nullable_float": 2.0, "nullable_int": 3.0}

    reordered = train[["label", "nullable_int", "nullable_float"]]
    expected = original.copy(deep=True)
    expected.loc[2, "nullable_float"] = 2.0
    result = imputer.transform(reordered)
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(reordered, train[reordered.columns])


def test_median_float64_special_values_match_pandas_without_mutating_input():
    import numpy as np

    frame = pd.DataFrame({
        "finite_nan": [1.0, np.nan, 3.0, 5.0],
        "positive_infinity": [1.0, 2.0, np.inf, np.nan],
        "negative_infinity": [-np.inf, -3.0, -1.0, np.nan],
        "opposite_infinities": [-np.inf, np.inf, np.nan, np.nan],
        "signed_zero": [-0.0, 0.0, np.nan, np.nan],
        "all_missing": [np.nan, np.nan, np.nan, np.nan],
    })
    before = frame.copy(deep=True)
    expected = frame.median().to_dict()
    imputer = MeanImputer(imputation_method="median", variables=list(frame))
    imputer.fit(frame)
    pd.testing.assert_frame_equal(frame, before)

    for name, value in expected.items():
        actual = imputer.imputer_dict_[name]
        if pd.isna(value):
            assert pd.isna(actual), name
        else:
            assert actual == value, name
            if value == 0.0:
                assert np.signbit(actual) == np.signbit(value), name


def test_median_nullable_fallback_matches_pandas_and_keeps_input_independent():
    frame = pd.DataFrame({
        "nullable_float": pd.array([1.0, pd.NA, 9.0, 20.0], dtype="Float64"),
        "nullable_int": pd.array([1, 3, pd.NA, 20], dtype="Int64"),
        "label": ["a", "b", "c", "d"],
    }, index=pd.Index([8, 2, 5, 1], name="row"))
    before = frame.copy(deep=True)
    variables = ["nullable_float", "nullable_int"]
    expected_medians = frame[variables].median().to_dict()
    imputer = MeanImputer(imputation_method="median", variables=variables)
    imputer.fit(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert imputer.imputer_dict_ == expected_medians

    expected = before.copy(deep=True)
    for name, value in expected_medians.items():
        expected[name] = expected[name].fillna(value)
    output = imputer.transform(frame)
    pd.testing.assert_frame_equal(output, expected)
    pd.testing.assert_frame_equal(frame, before)
    output.index.name = "output_row"
    assert frame.index.name == "row"
