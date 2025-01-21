import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine.discretisation import (
    ArbitraryDiscretiser,
    DecisionTreeDiscretiser,
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    GeometricWidthDiscretiser,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)


_estimators = [
    DecisionTreeDiscretiser(regression=False),
    EqualFrequencyDiscretiser(),
    EqualWidthDiscretiser(),
    ArbitraryDiscretiser(binning_dict={"x0": [-np.inf, 0, np.inf]}),
    GeometricWidthDiscretiser(),
]

if sklearn_version < parse_version("1.6"):

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(estimator)

else:

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(
            estimator=estimator,
            expected_failed_checks=estimator._more_tags()["_xfail_checks"],
        )


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    if estimator.__class__.__name__ == "ArbitraryDiscretiser":
        estimator.set_params(binning_dict={"var_1": [-np.inf, 0, np.inf]})
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_within_pipeline(transformer):
    if transformer.__class__.__name__ == "ArbitraryDiscretiser":
        transformer.set_params(binning_dict={"feature_1": [-np.inf, 0, np.inf]})

    X = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X, y)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)
