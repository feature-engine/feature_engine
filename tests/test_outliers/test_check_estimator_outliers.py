import pandas as pd
import pytest
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine.outliers import ArbitraryOutlierCapper, OutlierTrimmer, Winsorizer
from feature_engine.tags import _return_tags
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [
    ArbitraryOutlierCapper(max_capping_dict={"x0": 10}),
    OutlierTrimmer(),
    Winsorizer(),
]

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

if sklearn_version < parse_version("1.6"):

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(estimator)

else:
    FAILED_CHECKS = _return_tags()["_xfail_checks"]
    FAILED_CHECKS_AOC = _return_tags()["_xfail_checks"]

    msg1 = (
        "transformers raise errors when data variation is low, " "thus this check fails"
    )

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
        return check_estimator(estimator=estimator, expected_failed_checks=failed_tests)


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

    if transformer.__class__.__name__ == "Winsorizer":
        transformer.set_params(add_indicators=True)

        pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

        Xtt = transformer.fit_transform(X)
        Xtp = pipe.fit_transform(X, y)

        pd.testing.assert_frame_equal(Xtt, Xtp)
