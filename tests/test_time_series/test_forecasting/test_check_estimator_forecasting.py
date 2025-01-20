import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine.timeseries.forecasting import (
    ExpandingWindowFeatures,
    LagFeatures,
    WindowFeatures,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [
    LagFeatures(missing_values="ignore"),
    WindowFeatures(missing_values="ignore"),
    ExpandingWindowFeatures(missing_values="ignore"),
]


sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

if sklearn_version < parse_version("1.6"):

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(estimator)

else:
    lf = LagFeatures(missing_values="ignore")
    wf = WindowFeatures(missing_values="ignore")
    ewf = ExpandingWindowFeatures(missing_values="ignore")
    failing_checks = {
        "check_estimators_nan_inf": "Time Series transformers do not handle NaNs "
        "or infinity."
    }

    @pytest.mark.parametrize(
        "estimator, failed_tests",
        [
            (lf, {**failing_checks, **lf._more_tags()["_xfail_checks"]}),
            (wf, {**failing_checks, **wf._more_tags()["_xfail_checks"]}),
            (ewf, {**failing_checks, **ewf._more_tags()["_xfail_checks"]}),
        ],
    )
    def test_check_estimator_from_sklearn(estimator, failed_tests):
        return check_estimator(estimator=estimator, expected_failed_checks=failed_tests)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_error_when_not_unique_values_in_index(df_time, estimator):
    X = df_time.copy()

    # introduce dupes in index
    tmp = X.head(2).copy()
    tmp.iloc[0] = [1, 1, 1, "blue"]
    Xd = pd.concat([X, tmp], axis=0)

    transformer = clone(estimator)

    with pytest.raises(NotImplementedError):
        transformer.fit(Xd)

    transformer.fit(X)
    with pytest.raises(NotImplementedError):
        transformer.transform(Xd)


@pytest.mark.parametrize("estimator", _estimators)
def test_error_when_nan_in_index(df_time, estimator):
    X = df_time.copy()

    # Introduce NaN in index.
    tmp = X.head(1).copy()
    tmp.index = [np.nan]
    Xd = pd.concat([X, tmp], axis=0)

    transformer = clone(estimator)

    with pytest.raises(NotImplementedError):
        transformer.fit(Xd)

    transformer.fit(X)
    with pytest.raises(NotImplementedError):
        transformer.transform(Xd)


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_in_pipeline_with_set_output_pandas(df_time, transformer):
    X = df_time.copy()

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X)

    pd.testing.assert_frame_equal(Xtt, Xtp)
