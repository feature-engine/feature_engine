import pandas as pd
import pytest
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine.creation import (
    CyclicalFeatures,
    DecisionTreeFeatures,
    MathFeatures,
    RelativeFeatures,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

_estimators = [
    MathFeatures(variables=["x0", "x1"], func="mean", missing_values="ignore"),
    RelativeFeatures(
        variables=["x0", "x1"], reference=["x0"], func=["add"], missing_values="ignore"
    ),
    CyclicalFeatures(),
    DecisionTreeFeatures(regression=False),
]

if sklearn_version > parse_version("1.6"):

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(
            estimator=estimator,
            expected_failed_checks=estimator._more_tags()["_xfail_checks"],
        )

else:

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(estimator)


_estimators = [
    MathFeatures(variables=["var_1", "var_2", "var_3"], func="mean"),
    RelativeFeatures(variables=["var_1", "var_2"], reference=["var_3"], func=["add"]),
    CyclicalFeatures(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)


_estimators = [
    CyclicalFeatures(),
    MathFeatures(variables=["feature_1", "feature_2"], func=["sum", "mean"]),
    RelativeFeatures(variables=["feature_1"], reference=["feature_2"], func=["div"]),
]


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_in_pipeline_with_set_output_pandas(transformer):
    X = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)
