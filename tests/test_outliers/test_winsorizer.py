import math
import re

import pytest

from feature_engine.outliers import Winsoriser, Winsorizer
from tests.backend_helpers import to_dict

DEPRECATION_WARNING = (
    "Winsorizer was deprecated in favour of Winsoriser in version 2.0.0 and will "
    "be removed in version 2.1.0. To silence this warning, use Winsoriser instead."
)

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)

VARTYPES = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}


@pytest.fixture(
    params=[Winsoriser, Winsorizer],
    ids=["Winsoriser", "Winsorizer"],
)
def transformer_class(request):
    return request.param


def make_transformer(transformer_class, **kwargs):
    if transformer_class is Winsorizer:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            return transformer_class(**kwargs)
    return transformer_class(**kwargs)


def test_winsorizer_raises_future_warning():
    with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
        Winsorizer()


def test_gaussian_capping_right_tail_with_fold_1(
    make_df, data_normal_dist, transformer_class
):
    # test case 1: mean and std, right tail
    transformer = make_transformer(
        transformer_class, capping_method="gaussian", tail="right", fold=1
    )
    X_out = transformer.fit_transform(make_df(data_normal_dist))

    # test init params
    assert transformer.capping_method == "gaussian"
    assert transformer.tail == "right"
    assert transformer.fold == 1
    # test fit attr
    assert math.isclose(transformer.right_tail_caps_["var"], 0.1067690260251065)
    assert transformer.left_tail_caps_ == {}
    assert transformer.n_features_in_ == 1
    # test transform outputs
    assert isinstance(X_out, make_df)
    assert math.isclose(max(to_dict(X_out)["var"]), 0.1067690260251065)


def test_gaussian_capping_both_tails_with_fold_2(
    make_df, data_normal_dist, transformer_class
):
    # test case 2: mean and std, both tails, different fold value
    transformer = make_transformer(
        transformer_class, capping_method="gaussian", tail="both", fold=2
    )
    X_out = transformer.fit_transform(make_df(data_normal_dist))

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.2075572504967645)
    assert math.isclose(transformer.left_tail_caps_["var"], -0.1955956473898675)
    # test transform output
    assert isinstance(X_out, make_df)
    values = to_dict(X_out)["var"]
    assert math.isclose(max(values), 0.2075572504967645)
    assert math.isclose(min(values), -0.1955956473898675)


def test_iqr_capping_both_tails_with_fold_1(
    make_df, data_normal_dist, transformer_class
):
    # test case 3: IQR, both tails, fold 1
    transformer = make_transformer(
        transformer_class, capping_method="iqr", tail="both", fold=1
    )
    X_out = transformer.fit_transform(make_df(data_normal_dist))

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.21180113880445128)
    assert math.isclose(transformer.left_tail_caps_["var"], -0.20247907173293223)
    # test transform output
    assert isinstance(X_out, make_df)
    values = to_dict(X_out)["var"]
    assert math.isclose(max(values), 0.21180113880445128)
    assert math.isclose(min(values), -0.20247907173293223)


def test_iqr_capping_left_tail_with_fold_2(
    make_df, data_normal_dist, transformer_class
):
    # test case 4: IQR, left tail, fold 0.8
    transformer = make_transformer(
        transformer_class, capping_method="iqr", tail="left", fold=0.8
    )
    X_out = transformer.fit_transform(make_df(data_normal_dist))

    # test fit params
    assert transformer.right_tail_caps_ == {}
    assert math.isclose(transformer.left_tail_caps_["var"], -0.17486039103044)
    # test transform output
    assert isinstance(X_out, make_df)
    assert math.isclose(min(to_dict(X_out)["var"]), -0.17486039103044)


def test_quantile_capping_both_tails_with_fold_10_percent(
    make_df, data_normal_dist, transformer_class
):
    # test case 5: quantiles, both tails, fold 10%
    transformer = make_transformer(
        transformer_class, capping_method="quantiles", tail="both", fold=0.1
    )
    X_out = transformer.fit_transform(make_df(data_normal_dist))

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.14712481122898166)
    assert math.isclose(transformer.left_tail_caps_["var"], -0.12366227743232801)
    # test transform output
    assert isinstance(X_out, make_df)
    values = to_dict(X_out)["var"]
    assert math.isclose(max(values), 0.14712481122898166)
    assert math.isclose(min(values), -0.12366227743232801)


def test_quantile_capping_right_tail_with_fold_15_percent(
    make_df, data_normal_dist, transformer_class
):
    # test case 6: quantiles, right tail, fold 15%
    transformer = make_transformer(
        transformer_class, capping_method="quantiles", tail="right", fold=0.15
    )
    X_out = transformer.fit_transform(make_df(data_normal_dist))

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.11823196128033647)
    assert transformer.left_tail_caps_ == {}
    # test transform output
    assert isinstance(X_out, make_df)
    assert math.isclose(max(to_dict(X_out)["var"]), 0.11823196128033647)


@pytest.mark.parametrize(
    "strings,expected",
    [("gaussian", 3), ("iqr", 1.5), ("mad", 3.29), ("quantiles", 0.05)],
)
def test_auto_fold_default_value(
    make_df, data_normal_dist, strings, expected, transformer_class
):
    transformer = make_transformer(
        transformer_class, capping_method=strings, fold="auto"
    )
    transformer.fit(make_df(data_normal_dist))
    assert transformer.fold_ == expected


def test_mad_capping_right_tail_with_fold_1(
    make_df, data_normal_dist, transformer_class
):
    # test case: median and mad, right tail
    transformer = make_transformer(
        transformer_class, capping_method="mad", tail="right", fold=1
    )
    X_out = transformer.fit_transform(make_df(data_normal_dist))

    # test init params
    assert transformer.capping_method == "mad"
    assert transformer.tail == "right"
    assert transformer.fold == 1
    # test fit attr
    assert math.isclose(transformer.right_tail_caps_["var"], 0.10995521088494983)
    assert transformer.left_tail_caps_ == {}
    assert transformer.n_features_in_ == 1
    # test transform outputs
    assert isinstance(X_out, make_df)
    assert math.isclose(max(to_dict(X_out)["var"]), 0.10995521088494983)


def test_mad_capping_both_tails_with_fold_2(
    make_df, data_normal_dist, transformer_class
):
    # test case: mad, both tails, different fold value
    transformer = make_transformer(
        transformer_class, capping_method="mad", tail="both", fold=2
    )
    X_out = transformer.fit_transform(make_df(data_normal_dist))

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.21050080982609987)
    assert math.isclose(transformer.left_tail_caps_["var"], -0.1916815859385002)
    # test transform output
    assert isinstance(X_out, make_df)
    values = to_dict(X_out)["var"]
    assert math.isclose(max(values), 0.21050080982609987)
    assert math.isclose(min(values), -0.1916815859385002)


def test_indicators_are_added(make_df, data_normal_dist, transformer_class):
    X = make_df(data_normal_dist)
    n_cols = X.shape[1]

    transformer = make_transformer(
        transformer_class,
        tail="both",
        capping_method="quantiles",
        fold=0.1,
        add_indicators=True,
    )
    X_out = transformer.fit_transform(X)
    assert isinstance(X_out, make_df)
    assert X_out.shape[1] == 3 * n_cols
    result = to_dict(X_out)
    for col in list(X_out.columns)[n_cols:]:
        assert sum(result[col]) > 0

    transformer = make_transformer(
        transformer_class,
        tail="left",
        capping_method="quantiles",
        fold=0.1,
        add_indicators=True,
    )
    X_out = transformer.fit_transform(X)
    assert isinstance(X_out, make_df)
    assert X_out.shape[1] == 2 * n_cols
    result = to_dict(X_out)
    for col in list(X_out.columns)[n_cols:]:
        assert sum(result[col]) > 0

    transformer = make_transformer(
        transformer_class,
        tail="right",
        capping_method="quantiles",
        fold=0.1,
        add_indicators=True,
    )
    X_out = transformer.fit_transform(X)
    assert isinstance(X_out, make_df)
    assert X_out.shape[1] == 2 * n_cols
    result = to_dict(X_out)
    for col in list(X_out.columns)[n_cols:]:
        assert sum(result[col]) > 0


def test_indicators_filter_variables(make_df, transformer_class):
    X = make_df(VARTYPES)
    n_cols = X.shape[1]

    transformer = make_transformer(
        transformer_class,
        variables=["Age", "Marks"],
        tail="both",
        capping_method="quantiles",
        fold=0.1,
        add_indicators=True,
    )
    X_out = transformer.fit_transform(X)
    assert isinstance(X_out, make_df)
    assert X_out.shape[1] == n_cols + 4

    transformer.set_params(tail="left")
    X_out = transformer.fit_transform(X)
    assert X_out.shape[1] == n_cols + 2

    transformer.set_params(tail="right")
    X_out = transformer.fit_transform(X)
    assert X_out.shape[1] == n_cols + 2


def test_indicators_are_correct(make_df, transformer_class):
    X = make_df({"col": [float(i) for i in range(100)]})
    expected_left = [1.0] * 10 + [0.0] * 90
    expected_right = [0.0] * 90 + [1.0] * 10

    transformer = make_transformer(
        transformer_class,
        tail="left",
        capping_method="quantiles",
        fold=0.1,
        add_indicators=True,
    )
    X_out = transformer.fit_transform(X)
    assert isinstance(X_out, make_df)
    assert to_dict(X_out)["col_left"] == expected_left

    transformer.set_params(tail="right")
    X_out = transformer.fit_transform(X)
    assert to_dict(X_out)["col_right"] == expected_right

    transformer.set_params(tail="both")
    X_out = transformer.fit_transform(X)
    result = to_dict(X_out)
    assert result["col_left"] == expected_left
    assert result["col_right"] == expected_right
    assert list(X_out.columns) == ["col", "col_left", "col_right"]


def test_transformer_ignores_na_in_df(make_df, data_na, transformer_class):
    # test case: dataset contains na and transformer is asked to ignore them
    transformer = make_transformer(
        transformer_class,
        capping_method="gaussian",
        tail="right",
        fold=1,
        variables=["Age", "Marks"],
        missing_values="ignore",
    )
    X_out = transformer.fit_transform(make_df(data_na))

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["Age"], 38.04494616731882)
    assert math.isclose(transformer.right_tail_caps_["Marks"], 0.8784116651786605)
    assert transformer.left_tail_caps_ == {}
    assert transformer.n_features_in_ == 5
    # test transform output
    assert isinstance(X_out, make_df)
    result = to_dict(X_out)
    age = [v for v in result["Age"] if v is not None]
    marks = [v for v in result["Marks"] if v is not None]
    assert math.isclose(max(age), 38.04494616731882)
    assert math.isclose(max(marks), 0.8784116651786605)


def test_error_if_capping_method_not_permitted(transformer_class):
    with pytest.raises(ValueError):
        make_transformer(transformer_class, capping_method="other")


def test_error_if_tail_value_not_permitted(transformer_class):
    with pytest.raises(ValueError):
        make_transformer(transformer_class, tail="other")


def test_error_if_missing_values_not_permited(transformer_class):
    with pytest.raises(ValueError):
        make_transformer(transformer_class, missing_values="other")


def test_error_if_fold_value_not_permitted(transformer_class):
    with pytest.raises(ValueError):
        make_transformer(transformer_class, fold=-1)


def test_error_if_capping_method_quantiles_and_fold_value_not_permitted(
    transformer_class,
):
    with pytest.raises(ValueError):
        make_transformer(transformer_class, capping_method="quantiles", fold=0.3)


def test_error_if_add_incators_not_permitted(transformer_class):
    with pytest.raises(ValueError):
        make_transformer(transformer_class, add_indicators=-1)
    with pytest.raises(ValueError):
        make_transformer(transformer_class, add_indicators=())
    with pytest.raises(ValueError):
        make_transformer(transformer_class, add_indicators=[True])


def test_fit_raises_error_if_na_in_inut_df(make_df, data_na, transformer_class):
    # test case: when dataset contains na, fit method
    transformer = make_transformer(transformer_class)
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.fit(make_df(data_na))


def test_transform_raises_error_if_na_in_input_df(
    make_df, data_na, transformer_class
):
    # test case: when dataset contains na, transform method
    X_na = make_df({k: data_na[k] for k in ["Name", "City", "Age", "Marks"]})
    transformer = make_transformer(transformer_class)
    transformer.fit(make_df(VARTYPES))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(X_na)


def test_get_feature_names_out(make_df, data_na, transformer_class):
    X = make_df(data_na)
    original_features = list(data_na)
    input_features = ["Age", "Marks"]

    # when indicators is false, we've got the generic check.
    # We need to test only when true
    tr = make_transformer(
        transformer_class,
        tail="left",
        add_indicators=True,
        missing_values="ignore",
    )
    tr.fit(X)

    out = [f + "_left" for f in input_features]
    assert tr.get_feature_names_out() == original_features + out
    assert tr.get_feature_names_out(original_features) == original_features + out

    tr = make_transformer(
        transformer_class,
        tail="right",
        add_indicators=True,
        missing_values="ignore",
    )
    tr.fit(X)

    out = [f + "_right" for f in input_features]
    assert tr.get_feature_names_out() == original_features + out
    assert tr.get_feature_names_out(original_features) == original_features + out

    tr = make_transformer(
        transformer_class,
        tail="both",
        add_indicators=True,
        missing_values="ignore",
    )
    tr.fit(X)

    out = ["Age_left", "Age_right", "Marks_left", "Marks_right"]
    assert tr.get_feature_names_out() == original_features + out
    assert tr.get_feature_names_out(original_features) == original_features + out


def test_low_variation(make_df, data_normal_dist, transformer_class):
    X = make_df({"var": [v // 10 for v in data_normal_dist["var"]]})
    transformer = make_transformer(transformer_class, capping_method="mad")
    with pytest.raises(ValueError, match="have low variation for method 'mad'"):
        transformer.fit(X)
