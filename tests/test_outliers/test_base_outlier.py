import re

import narwhals as nw
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.outliers.base_outlier import BaseOutlier, WinsorizerBase
from tests.backend_helpers import frame_to_dict

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)


# init parameters
@pytest.mark.parametrize(
    "capping_method", ["arbitrary", "Gaussian", "", 1, None, ["iqr"]]
)
def test_error_if_capping_method_not_permitted(capping_method):
    msg = (
        "capping_method must be 'gaussian', 'iqr', 'mad', 'quantiles'. "
        f"Got {capping_method} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        WinsorizerBase(capping_method=capping_method)


@pytest.mark.parametrize("tail", ["other", "Right", "", 1, None, ["right"]])
def test_error_if_tail_not_permitted(tail):
    msg = f"tail must be 'right', 'left' or 'both'. Got {tail} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        WinsorizerBase(tail=tail)


@pytest.mark.parametrize("fold", ["other", "Auto", 0, -1, -0.5])
def test_error_if_fold_not_permitted(fold):
    msg = f"fold must be a positive number or 'auto'. Got {fold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        WinsorizerBase(fold=fold)


@pytest.mark.parametrize("fold", [0.3, 1, 5])
def test_error_if_fold_above_0_2_with_quantiles(fold):
    msg = (
        "with capping_method ='quantiles', fold takes values between 0 and "
        "0.20 only."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        WinsorizerBase(capping_method="quantiles", fold=fold)


@pytest.mark.parametrize("missing_values", ["other", "Raise", 1, True, None])
def test_error_if_missing_values_not_permitted(missing_values):
    msg = (
        "missing_values must be 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        WinsorizerBase(missing_values=missing_values)


@pytest.mark.parametrize(
    "capping_method, tail, fold, missing_values",
    [
        ("gaussian", "right", "auto", "raise"),
        ("iqr", "left", 2, "ignore"),
        ("mad", "both", 1.5, "raise"),
        ("quantiles", "both", 0.1, "ignore"),
    ],
)
def test_init_param_assignment(capping_method, tail, fold, missing_values):
    transformer = WinsorizerBase(
        capping_method=capping_method,
        tail=tail,
        fold=fold,
        missing_values=missing_values,
    )
    assert transformer.capping_method == capping_method
    assert transformer.tail == tail
    assert transformer.fold == fold
    assert transformer.missing_values == missing_values


# fit and transform
def _expected_caps(values, capping_method, fold):
    # reference limits computed with pandas
    s = pd.Series(values)
    if capping_method == "gaussian":
        return s.mean() + fold * s.std(ddof=0), s.mean() - fold * s.std(ddof=0)
    if capping_method == "iqr":
        iqr = s.quantile(0.75) - s.quantile(0.25)
        return s.quantile(0.75) + fold * iqr, s.quantile(0.25) - fold * iqr
    if capping_method == "mad":
        mad = (s - s.median()).abs().median() / 0.67449
        return s.median() + fold * mad, s.median() - fold * mad
    return s.quantile(1 - fold), s.quantile(fold)


@pytest.mark.parametrize(
    "capping_method, fold",
    [("gaussian", 3), ("gaussian", 1), ("iqr", 1.5), ("mad", 2), ("quantiles", 0.1)],
)
def test_fit_learns_caps(make_df, data_normal_dist, capping_method, fold):
    transformer = WinsorizerBase(capping_method=capping_method, tail="both", fold=fold)
    transformer.fit(make_df(data_normal_dist))

    right, left = _expected_caps(data_normal_dist["var"], capping_method, fold)
    assert transformer.right_tail_caps_ == {"var": pytest.approx(right)}
    assert transformer.left_tail_caps_ == {"var": pytest.approx(left)}
    assert transformer.variables_ == ["var"]
    assert transformer.feature_names_in_ == ["var"]
    assert transformer.n_features_in_ == 1


@pytest.mark.parametrize("tail", ["right", "left"])
def test_fit_learns_caps_for_one_tail(make_df, data_normal_dist, tail):
    transformer = WinsorizerBase(tail=tail, fold=3).fit(make_df(data_normal_dist))

    right, left = _expected_caps(data_normal_dist["var"], "gaussian", 3)
    if tail == "right":
        assert transformer.right_tail_caps_ == {"var": pytest.approx(right)}
        assert transformer.left_tail_caps_ == {}
    else:
        assert transformer.left_tail_caps_ == {"var": pytest.approx(left)}
        assert transformer.right_tail_caps_ == {}


@pytest.mark.parametrize(
    "capping_method, expected",
    [("gaussian", 3.0), ("iqr", 1.5), ("mad", 3.29), ("quantiles", 0.05)],
)
def test_auto_fold(make_df, data_normal_dist, capping_method, expected):
    transformer = WinsorizerBase(capping_method=capping_method, fold="auto")
    transformer.fit(make_df(data_normal_dist))
    assert transformer.fold_ == expected


def test_fold_is_kept_when_given(make_df, data_normal_dist):
    transformer = WinsorizerBase(fold=2.5).fit(make_df(data_normal_dist))
    assert transformer.fold_ == 2.5


def test_fit_selects_numerical_variables_and_ignores_na(make_df, data_na):
    transformer = WinsorizerBase(tail="both", fold=1, missing_values="ignore")
    transformer.fit(make_df(data_na))

    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.feature_names_in_ == list(data_na)
    assert transformer.n_features_in_ == 5
    # missing values are skipped when learning the caps
    for var in ["Age", "Marks"]:
        values = [v for v in data_na[var] if v is not None]
        right, left = _expected_caps(values, "gaussian", 1)
        assert transformer.right_tail_caps_[var] == pytest.approx(right)
        assert transformer.left_tail_caps_[var] == pytest.approx(left)


def test_fit_raises_error_if_na(make_df, data_na):
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        WinsorizerBase().fit(make_df(data_na))


@pytest.mark.parametrize("capping_method", ["gaussian", "iqr", "mad", "quantiles"])
def test_error_if_low_variation(make_df, capping_method):
    X = make_df({"var": [1.0] * 10, "other": list(range(10))})
    msg = (
        f"Input columns ['var'] have low variation for method '{capping_method}'. "
        "Try other capping methods or drop these columns."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        WinsorizerBase(capping_method=capping_method, variables=["var"]).fit(X)


def test_fit_with_integer_column_names(data_normal_dist):
    # integer column names are pandas-only
    X = pd.DataFrame({0: data_normal_dist["var"]})
    transformer = WinsorizerBase(tail="both", fold=3).fit(X)

    right, left = _expected_caps(data_normal_dist["var"], "gaussian", 3)
    assert transformer.right_tail_caps_ == {0: pytest.approx(right)}
    assert transformer.left_tail_caps_ == {0: pytest.approx(left)}


class MockCapper(BaseOutlier):
    # caps are set by hand to test the shared transform logic
    def __init__(self, missing_values="raise"):
        self.missing_values = missing_values

    def fit(self, X, y=None):
        self.variables_ = ["a", "b", "c"]
        self.right_tail_caps_ = {"a": 2, "b": 2.5}
        self.left_tail_caps_ = {"a": 0, "c": 1}
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return self._transform(X)


DATA_CAP = {
    "a": [-1.0, 1.0, 3.0],
    "b": [1.0, 2.0, 3.0],
    "c": [0, 1, 2],
    "d": ["x", "y", "z"],
}


def test_transform_caps_values(make_df):
    X = make_df(DATA_CAP)
    Xt = MockCapper().fit(X).transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "a": [0.0, 1.0, 2.0],
        "b": [1.0, 2.0, 2.5],
        "c": [1, 1, 2],
        "d": ["x", "y", "z"],
    }
    # capping with a left bound only keeps integer columns as integers
    assert nw.from_native(Xt, eager_only=True)["c"].dtype.is_integer()


def test_transform_reorders_columns_to_match_fit(make_df):
    transformer = MockCapper().fit(make_df(DATA_CAP))
    reordered = make_df({k: DATA_CAP[k] for k in ["d", "c", "b", "a"]})

    Xt = transformer.transform(reordered)

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["a", "b", "c", "d"]


def test_transform_raises_error_if_different_number_of_columns(make_df):
    transformer = MockCapper().fit(make_df(DATA_CAP))
    msg = (
        "The number of columns in this dataset is different from the one used to "
        "fit this transformer (when using the fit() method)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.transform(make_df({k: DATA_CAP[k] for k in ["a", "b", "c"]}))


def test_transform_raises_error_if_na(make_df):
    transformer = MockCapper().fit(make_df(DATA_CAP))
    X_na = make_df({**DATA_CAP, "a": [-1.0, None, 3.0]})
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(X_na)


def test_transform_keeps_na_when_ignored(make_df):
    X_na = make_df({**DATA_CAP, "a": [-1.0, None, 3.0]})
    Xt = MockCapper(missing_values="ignore").fit(X_na).transform(X_na)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt)["a"] == [0.0, None, 2.0]


def test_transform_raises_non_fitted_error(make_df):
    msg = (
        "This MockCapper instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        MockCapper().transform(make_df(DATA_CAP))
