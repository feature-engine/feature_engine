import re

import numpy as np
import pytest

from feature_engine.outliers import Winsoriser, Winsorizer
from tests.backend_helpers import frame_to_dict

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


# init parameters
# the errors of the parameters from WinsorizerBase are tested in test_base_outlier.py
def test_winsorizer_raises_future_warning():
    with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
        Winsorizer()


@pytest.mark.parametrize("add_indicators", [-1, 1, "True", None, (), [True]])
def test_error_if_add_indicators_not_permitted(add_indicators, transformer_class):
    msg = (
        "add_indicators takes only booleans True and False. "
        f"Got {add_indicators} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        make_transformer(transformer_class, add_indicators=add_indicators)


@pytest.mark.parametrize(
    "capping_method, tail, fold, add_indicators, missing_values",
    [
        ("gaussian", "right", "auto", False, "raise"),
        ("iqr", "left", 2, True, "ignore"),
        ("mad", "both", 1.5, False, "ignore"),
        ("quantiles", "both", 0.1, True, "raise"),
    ],
)
def test_init_param_assignment(
    capping_method, tail, fold, add_indicators, missing_values, transformer_class
):
    transformer = make_transformer(
        transformer_class,
        capping_method=capping_method,
        tail=tail,
        fold=fold,
        add_indicators=add_indicators,
        missing_values=missing_values,
    )
    assert transformer.capping_method == capping_method
    assert transformer.tail == tail
    assert transformer.fold == fold
    assert transformer.add_indicators == add_indicators
    assert transformer.missing_values == missing_values


# fit and transform
@pytest.mark.parametrize(
    "capping_method, tail, fold, right, left",
    [
        ("gaussian", "right", 1, 0.1067690260251065, None),
        ("gaussian", "both", 2, 0.2075572504967645, -0.1955956473898675),
        ("iqr", "both", 1, 0.21180113880445128, -0.20247907173293223),
        ("iqr", "left", 0.8, None, -0.17486039103044),
        ("quantiles", "both", 0.1, 0.14712481122898166, -0.12366227743232801),
        ("quantiles", "right", 0.15, 0.11823196128033647, None),
        ("mad", "right", 1, 0.10995521088494983, None),
        ("mad", "both", 2, 0.21050080982609987, -0.1916815859385002),
    ],
)
def test_capping(
    make_df,
    data_normal_dist,
    transformer_class,
    capping_method,
    tail,
    fold,
    right,
    left,
):
    transformer = make_transformer(
        transformer_class, capping_method=capping_method, tail=tail, fold=fold
    )
    X_out = transformer.fit_transform(make_df(data_normal_dist))

    upper = np.inf if right is None else right
    lower = -np.inf if left is None else left
    expected = [min(max(v, lower), upper) for v in data_normal_dist["var"]]

    if right is None:
        assert transformer.right_tail_caps_ == {}
    else:
        assert transformer.right_tail_caps_ == {"var": pytest.approx(right)}
    if left is None:
        assert transformer.left_tail_caps_ == {}
    else:
        assert transformer.left_tail_caps_ == {"var": pytest.approx(left)}
    assert transformer.n_features_in_ == 1
    assert isinstance(X_out, make_df)
    assert frame_to_dict(X_out) == {"var": pytest.approx(expected)}


@pytest.mark.parametrize(
    "capping_method, expected",
    [("gaussian", 3), ("iqr", 1.5), ("mad", 3.29), ("quantiles", 0.05)],
)
def test_auto_fold_default_value(
    make_df, data_normal_dist, capping_method, expected, transformer_class
):
    transformer = make_transformer(
        transformer_class, capping_method=capping_method, fold="auto"
    )
    transformer.fit(make_df(data_normal_dist))
    assert transformer.fold_ == expected


@pytest.mark.parametrize("tail, n_indicators", [("both", 2), ("left", 1), ("right", 1)])
def test_indicators_are_added(
    make_df, data_normal_dist, transformer_class, tail, n_indicators
):
    X = make_df(data_normal_dist)
    transformer = make_transformer(
        transformer_class,
        tail=tail,
        capping_method="quantiles",
        fold=0.1,
        add_indicators=True,
    )
    X_out = transformer.fit_transform(X)

    assert isinstance(X_out, make_df)
    assert X_out.shape[1] == 1 + n_indicators
    result = frame_to_dict(X_out)
    for col in list(X_out.columns)[1:]:
        assert sum(result[col]) > 0


@pytest.mark.parametrize("tail, n_indicators", [("both", 4), ("left", 2), ("right", 2)])
def test_indicators_filter_variables(make_df, transformer_class, tail, n_indicators):
    X = make_df(VARTYPES)
    transformer = make_transformer(
        transformer_class,
        variables=["Age", "Marks"],
        tail=tail,
        capping_method="quantiles",
        fold=0.1,
        add_indicators=True,
    )
    X_out = transformer.fit_transform(X)

    assert isinstance(X_out, make_df)
    assert X_out.shape[1] == len(VARTYPES) + n_indicators


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
    assert frame_to_dict(X_out)["col_left"] == expected_left

    transformer.set_params(tail="right")
    X_out = transformer.fit_transform(X)
    assert frame_to_dict(X_out)["col_right"] == expected_right

    transformer.set_params(tail="both")
    X_out = transformer.fit_transform(X)
    result = frame_to_dict(X_out)
    assert result["col_left"] == expected_left
    assert result["col_right"] == expected_right
    assert list(X_out.columns) == ["col", "col_left", "col_right"]


def test_transformer_ignores_na_in_df(make_df, data_na, transformer_class):
    transformer = make_transformer(
        transformer_class,
        capping_method="gaussian",
        tail="right",
        fold=1,
        variables=["Age", "Marks"],
        missing_values="ignore",
    )
    X_out = transformer.fit_transform(make_df(data_na))

    assert transformer.right_tail_caps_ == {
        "Age": pytest.approx(38.04494616731882),
        "Marks": pytest.approx(0.8784116651786605),
    }
    assert transformer.left_tail_caps_ == {}
    assert transformer.n_features_in_ == 5
    assert isinstance(X_out, make_df)
    result = frame_to_dict(X_out)
    for var, cap in [("Age", 38.04494616731882), ("Marks", 0.8784116651786605)]:
        expected = [None if v is None else min(v, cap) for v in data_na[var]]
        assert result[var] == pytest.approx(expected)


def test_fit_raises_error_if_na_in_input_df(make_df, data_na, transformer_class):
    transformer = make_transformer(transformer_class)
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.fit(make_df(data_na))


def test_transform_raises_error_if_na_in_input_df(
    make_df, data_na, transformer_class
):
    X_na = make_df({k: data_na[k] for k in ["Name", "City", "Age", "Marks"]})
    transformer = make_transformer(transformer_class)
    transformer.fit(make_df(VARTYPES))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(X_na)


# without indicators, the feature names are covered by the generic checks
@pytest.mark.parametrize(
    "tail, indicators",
    [
        ("left", ["Age_left", "Marks_left"]),
        ("right", ["Age_right", "Marks_right"]),
        ("both", ["Age_left", "Age_right", "Marks_left", "Marks_right"]),
    ],
)
def test_get_feature_names_out(make_df, data_na, transformer_class, tail, indicators):
    original_features = list(data_na)
    tr = make_transformer(
        transformer_class, tail=tail, add_indicators=True, missing_values="ignore"
    )
    tr.fit(make_df(data_na))

    expected = original_features + indicators
    assert tr.get_feature_names_out() == expected
    assert tr.get_feature_names_out(original_features) == expected


def test_variables_without_variation_are_left_untouched(
    make_df, data_normal_dist, transformer_class
):
    data = {
        "var": [v // 10 for v in data_normal_dist["var"]],
        "other": data_normal_dist["var"],
    }
    transformer = make_transformer(
        transformer_class, capping_method="mad", tail="both", add_indicators=True
    )
    Xt = transformer.fit_transform(make_df(data))

    assert transformer.right_tail_caps_["var"] == np.inf
    assert transformer.left_tail_caps_["var"] == -np.inf
    assert isinstance(Xt, make_df)
    result = frame_to_dict(Xt)
    assert result["var"] == data["var"]
    assert result["var_left"] == [0.0] * len(data["var"])
    assert result["var_right"] == [0.0] * len(data["var"])
