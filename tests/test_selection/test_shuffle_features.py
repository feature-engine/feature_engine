import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import SelectByShuffling
from tests.backend_helpers import frame_to_dict, make_series

VARIABLES = ["var_0", "var_4", "var_7", "var_9"]


def _split_target(make_df, data):
    X = make_df({k: v for k, v in data.items() if k != "target"})
    return X, make_series(make_df, data["target"])


def _random_forest():
    return RandomForestClassifier(n_estimators=10, random_state=1)


@pytest.fixture(scope="module")
def data_diabetes():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    data = {col: X[col].tolist() for col in X.columns}
    data["target"] = y.tolist()
    return data


# init parameters
@pytest.mark.parametrize("threshold", ["hello", "", [0.1], [], {"a": 1}, (1,)])
def test_error_if_threshold_not_number_or_none(threshold):
    msg = f"threshold must be an integer, a float or None. Got {threshold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectByShuffling(_random_forest(), threshold=threshold)


@pytest.mark.parametrize("confirm_variables", ["True", 1, None, [True]])
def test_error_if_confirm_variables_not_bool(confirm_variables):
    msg = (
        "confirm_variables takes only values True and False. "
        f"Got {confirm_variables} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectByShuffling(_random_forest(), confirm_variables=confirm_variables)


@pytest.mark.parametrize(
    "estimator, scoring, cv, threshold, random_state, confirm_variables",
    [
        (RandomForestClassifier(), "roc_auc", 3, None, None, False),
        (LogisticRegression(), "accuracy", StratifiedKFold(), 0.01, 1, True),
        (LinearRegression(), "r2", KFold(2), 5, 10, False),
    ],
)
def test_init_param_assignment(
    estimator, scoring, cv, threshold, random_state, confirm_variables
):
    sel = SelectByShuffling(
        estimator,
        scoring=scoring,
        cv=cv,
        threshold=threshold,
        random_state=random_state,
        confirm_variables=confirm_variables,
    )
    assert sel.estimator is estimator
    assert sel.scoring == scoring
    assert sel.cv is cv
    assert sel.threshold == threshold
    assert sel.random_state == random_state
    assert sel.confirm_variables is confirm_variables


# fit and transform
def test_fit_attributes(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = SelectByShuffling(
        _random_forest(), variables=VARIABLES, threshold=0.01, random_state=1
    )
    sel.fit(X, y)

    assert sel.initial_model_performance_ == pytest.approx(0.9965735235313549)
    assert sel.performance_drifts_ == pytest.approx(
        {
            "var_0": 0.0037026895460631204,
            "var_4": 0.0017710452198405058,
            "var_7": 0.25884577270119447,
            "var_9": -1.959497441428315e-05,
        }
    )
    assert sel.performance_drifts_std_ == pytest.approx(
        {
            "var_0": 0.0005724303014393721,
            "var_4": 0.0005148668118348698,
            "var_7": 0.02708631411106102,
            "var_9": 0.0016362909674612345,
        }
    )
    assert sel.features_to_drop_ == ["var_0", "var_4", "var_9"]
    assert sel.variables_ == VARIABLES
    assert sel.feature_names_in_ == [f"var_{i}" for i in range(12)]
    assert sel.n_features_in_ == 12


def test_transform_removes_features_below_threshold(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = SelectByShuffling(_random_forest(), threshold=0.01, random_state=1)
    Xt = sel.fit_transform(X, y)

    assert sel.initial_model_performance_ == pytest.approx(0.9953593232960704)
    assert sel.features_to_drop_ == [
        f"var_{i}" for i in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    ]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var_7": data_classification["var_7"]}


def test_threshold_none_uses_mean_drift(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = SelectByShuffling(
        LogisticRegression(), scoring="accuracy", variables=VARIABLES, random_state=3
    )
    sel.fit(X, y)

    # the mean drift is 0.13, only var_7 is above it.
    assert sel.performance_drifts_ == pytest.approx(
        {
            "var_0": 0.021998045950141765,
            "var_4": 0.005002007996020019,
            "var_7": 0.4890009770249292,
            "var_9": 0.004001006995019041,
        }
    )
    assert sel.features_to_drop_ == ["var_0", "var_4", "var_9"]


def test_regression_with_r2(make_df, data_diabetes):
    X, y = _split_target(make_df, data_diabetes)
    sel = SelectByShuffling(
        LinearRegression(), scoring="r2", cv=3, threshold=0.05, random_state=1
    )
    Xt = sel.fit_transform(X, y)

    assert sel.initial_model_performance_ == pytest.approx(0.48870212980353145)
    assert sel.performance_drifts_ == pytest.approx(
        {
            "age": 0.0019221800216577822,
            "sex": 0.058082587710752975,
            "bmi": 0.16770566452186242,
            "bp": 0.0702676565845552,
            "s1": 0.5151999913097192,
            "s2": 0.17198756874519755,
            "s3": 0.01669668794929896,
            "s4": 0.025518182049075577,
            "s5": 0.4065911011104548,
            "s6": 0.0038028879119369474,
        }
    )
    assert sel.features_to_drop_ == ["age", "s3", "s4", "s6"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["sex", "bmi", "bp", "s1", "s2", "s5"]


def test_regression_with_neg_mse_and_target_as_dataframe(make_df, data_diabetes):
    X, _ = _split_target(make_df, data_diabetes)
    y = make_df({"target": data_diabetes["target"]})
    sel = SelectByShuffling(
        DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=1000,
        random_state=1,
    )
    sel.fit(X, y)

    assert sel.initial_model_performance_ == pytest.approx(-5835.58371040724)
    assert sel.performance_drifts_ == pytest.approx(
        {
            "age": -227.27149321266916,
            "sex": -78.94570135746653,
            "bmi": 2132.739819004524,
            "bp": 134.5814479638011,
            "s1": 279.30316742081413,
            "s2": 313.13800904977325,
            "s3": 19.719457013574356,
            "s4": 2050.5294117647063,
            "s5": 1661.9977375565613,
            "s6": -4.617647058823422,
        }
    )
    assert sel.features_to_drop_ == ["age", "sex", "bp", "s1", "s2", "s3", "s6"]


@pytest.mark.parametrize("cv_type", ["int", "splitter", "generator"])
def test_cv_options(make_df, data_classification, cv_type):
    X, y = _split_target(make_df, data_classification)
    cv = {"int": 3, "splitter": StratifiedKFold(n_splits=3)}.get(cv_type)
    if cv_type == "generator":
        cv = StratifiedKFold(n_splits=3).split(X, y)

    sel = SelectByShuffling(
        _random_forest(), variables=VARIABLES, cv=cv, threshold=0.01, random_state=1
    )
    sel.fit(X, y)

    assert sel.initial_model_performance_ == pytest.approx(0.9965735235313549)
    assert sel.features_to_drop_ == ["var_0", "var_4", "var_9"]


class _FoldsChangeOnEachCall(KFold):
    """Splitter that yields different folds every time split() is called."""

    def split(self, X, y=None, groups=None):
        self.n_calls = getattr(self, "n_calls", 0) + 1
        folds = list(super().split(X, y, groups))
        return iter(folds if self.n_calls % 2 == 1 else folds[::-1])


def test_performance_is_evaluated_on_the_folds_of_each_model(
    make_df, data_classification
):
    # the shuffled data must be scored on the fold each model was not trained on.
    X, y = _split_target(make_df, data_classification)
    params = dict(variables=VARIABLES, random_state=1)
    sel = SelectByShuffling(
        _random_forest(), cv=_FoldsChangeOnEachCall(n_splits=3), **params
    )
    sel.fit(X, y)
    reference = SelectByShuffling(_random_forest(), cv=KFold(n_splits=3), **params)
    reference.fit(X, y)

    assert sel.performance_drifts_ == pytest.approx(reference.performance_drifts_)
    assert sel.performance_drifts_std_ == pytest.approx(
        reference.performance_drifts_std_
    )


def test_non_numerical_variables_are_ignored(make_df, data_classification):
    data = {**data_classification, "cat_1": ["a"] * 1000, "cat_2": ["b"] * 1000}
    X, y = _split_target(make_df, data)
    sel = SelectByShuffling(_random_forest(), threshold=0.01, random_state=1)
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == [f"var_{i}" for i in range(12)]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_7": data_classification["var_7"],
        "cat_1": ["a"] * 1000,
        "cat_2": ["b"] * 1000,
    }


def test_confirm_variables(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = SelectByShuffling(
        _random_forest(),
        variables=["var_0", "var_4", "var_7", "not_in_X"],
        confirm_variables=True,
        random_state=1,
    )
    sel.fit(X, y)

    assert sel.variables_ == ["var_0", "var_4", "var_7"]
    assert sel.features_to_drop_ == ["var_0", "var_4"]


def test_missing_values(make_df, data_classification):
    data = {k: data_classification[k] for k in [*VARIABLES, "target"]}
    data["var_0"] = [None if i % 7 == 0 else v for i, v in enumerate(data["var_0"])]
    data["var_7"] = [None if i % 5 == 0 else v for i, v in enumerate(data["var_7"])]
    X, y = _split_target(make_df, data)
    sel = SelectByShuffling(
        HistGradientBoostingClassifier(max_iter=10, random_state=0), random_state=1
    )
    sel.fit(X, y)

    assert sel.initial_model_performance_ == pytest.approx(0.993134587411696)
    assert sel.performance_drifts_ == pytest.approx(
        {
            "var_0": 0.008290557612846916,
            "var_4": 0.35213096252252896,
            "var_7": 0.002123827199128625,
            "var_9": 0.000764131562324577,
        }
    )
    assert sel.features_to_drop_ == ["var_0", "var_7", "var_9"]


def test_sample_weight(make_df):
    X = make_df(
        {
            "x1": [1000, 2000, 1000, 1000, 2000, 3000],
            "x2": [1000, 2000, 1000, 1000, 2000, 3000],
        }
    )
    y = make_series(make_df, [1, 0, 0, 1, 1, 0])
    sel = SelectByShuffling(
        RandomForestClassifier(random_state=42), cv=2, random_state=42
    )
    sel.fit(X, y, sample_weight=[1000, 2000, 1000, 1000, 2000, 3000])

    assert sel.initial_model_performance_ == 0.125
    assert sel.features_to_drop_ == ["x2"]


@pytest.mark.parametrize("to_target", [list, np.array])
def test_target_as_list_and_array(make_df, data_classification, to_target):
    X, _ = _split_target(make_df, data_classification)
    y = to_target(data_classification["target"])
    sel = SelectByShuffling(
        _random_forest(), variables=VARIABLES, threshold=0.01, random_state=1
    )
    sel.fit(X, y)

    assert sel.initial_model_performance_ == pytest.approx(0.9965735235313549)
    assert sel.features_to_drop_ == ["var_0", "var_4", "var_9"]


def test_input_dataframe_is_not_modified(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    SelectByShuffling(_random_forest(), variables=VARIABLES, random_state=1).fit(X, y)

    assert frame_to_dict(X) == {
        k: v for k, v in data_classification.items() if k != "target"
    }


def test_integer_column_names_pandas(data_classification):
    X = pd.DataFrame({i: data_classification[f"var_{i}"] for i in range(12)})
    y = pd.Series(data_classification["target"])
    sel = SelectByShuffling(
        _random_forest(), variables=[0, 4, 7, 9], threshold=0.01, random_state=1
    )
    Xt = sel.fit_transform(X, y)

    assert sel.performance_drifts_ == pytest.approx(
        {
            0: 0.0037026895460631204,
            4: 0.0017710452198405058,
            7: 0.25884577270119447,
            9: -1.959497441428315e-05,
        }
    )
    assert sel.features_to_drop_ == [0, 4, 9]
    pd.testing.assert_frame_equal(Xt, X.drop(columns=[0, 4, 9]))


def test_non_default_index_pandas(data_classification):
    index = np.arange(1000)[::-1] + 50
    X = pd.DataFrame(
        {k: v for k, v in data_classification.items() if k != "target"}, index=index
    )
    y = pd.Series(data_classification["target"], index=index)
    sel = SelectByShuffling(
        _random_forest(), variables=VARIABLES, threshold=0.01, random_state=1
    )
    Xt = sel.fit_transform(X, y)

    assert sel.initial_model_performance_ == pytest.approx(0.9965735235313549)
    assert sel.features_to_drop_ == ["var_0", "var_4", "var_9"]
    pd.testing.assert_frame_equal(Xt, X.drop(columns=["var_0", "var_4", "var_9"]))


def test_nullable_integer_dtype_pandas(data_classification):
    X = pd.DataFrame({k: data_classification[k] for k in VARIABLES})
    X["var_0"] = pd.array(
        [None if i % 7 == 0 else round(v * 10) for i, v in enumerate(X["var_0"])],
        dtype="Int64",
    )
    y = pd.Series(data_classification["target"])
    sel = SelectByShuffling(
        HistGradientBoostingClassifier(max_iter=10, random_state=0), random_state=1
    )
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == VARIABLES
    assert sel.performance_drifts_ == pytest.approx(
        {
            "var_0": 0.0007341414720931638,
            "var_4": -0.00020804719599920585,
            "var_7": 0.45245947715827217,
            "var_9": 0.0001250311491274303,
        }
    )
    assert sel.features_to_drop_ == ["var_0", "var_4", "var_9"]
    pd.testing.assert_frame_equal(Xt, X[["var_7"]])
