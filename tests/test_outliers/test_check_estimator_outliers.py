import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.outliers import ArbitraryOutlierCapper, OutlierTrimmer, Winsoriser
from feature_engine.tags import _return_tags
from tests.backend_helpers import frame_to_dict
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from tests.estimator_checks.sklearn_check_wrapper import wrap_for_check_estimator

_estimators = [
    ArbitraryOutlierCapper(max_capping_dict={"x0": 10}),
    OutlierTrimmer(),
    Winsoriser(),
]

FAILED_CHECKS = _return_tags()["_xfail_checks"]
FAILED_CHECKS_AOC = _return_tags()["_xfail_checks"]

msg1 = "transformers raise errors when data variation is low, " "thus this check fails"

msg2 = "transformer has 1 mandatory parameter"

FAILED_CHECKS.update({"check_fit2d_1sample": msg1})
FAILED_CHECKS_AOC.update(
    {
        "check_fit2d_1sample": msg1,
        "check_parameters_default_constructible": msg2,
    }
)


@pytest.mark.parametrize(
    "estimator, failed_tests",
    [
        (_estimators[0], FAILED_CHECKS_AOC),
        (_estimators[1], FAILED_CHECKS),
        (_estimators[2], FAILED_CHECKS),
    ],
)
def test_check_estimator_from_sklearn(estimator, failed_tests):
    return check_estimator(
        estimator=wrap_for_check_estimator(estimator),
        expected_failed_checks=failed_tests,
    )


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    if estimator.__class__.__name__ == "ArbitraryOutlierCapper":
        estimator.set_params(max_capping_dict={"var_1": 10})
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_in_pipeline_with_set_output_pandas(transformer):
    if transformer.__class__.__name__ == "ArbitraryOutlierCapper":
        transformer.set_params(max_capping_dict={"feature_1": 10})

    X = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)

    if transformer.__class__.__name__ == "Winsoriser":
        transformer.set_params(add_indicators=True)

        pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

        Xtt = transformer.fit_transform(X)
        Xtp = pipe.fit_transform(X, y)

        pd.testing.assert_frame_equal(Xtt, Xtp)


_REORDER_DATA = {
    "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
    "b": [-50.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    "s": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"],
}


@pytest.mark.parametrize(
    "transformer",
    [
        ArbitraryOutlierCapper(max_capping_dict={"a": 5}, min_capping_dict={"b": 0}),
        Winsoriser(capping_method="iqr", tail="both", add_indicators=True),
        OutlierTrimmer(capping_method="iqr", tail="both"),
    ],
)
def test_transform_reorders_columns_as_in_train_set(make_df, transformer):
    X = make_df(_REORDER_DATA)
    expected = transformer.fit(X).transform(X)

    reordered = ["s", "b", "a"]
    Xt = transformer.transform(make_df({var: _REORDER_DATA[var] for var in reordered}))

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == list(expected.columns)
    assert frame_to_dict(Xt) == frame_to_dict(expected)
