import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.transformation import (
    ArcsinTransformer,
    ArcSinhTransformer,
    BoxCoxTransformer,
    LogCpTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from tests.estimator_checks.non_fitted_error_checks import (
    check_raises_non_fitted_error_when_fit_fails,
)

_estimators = [
    BoxCoxTransformer(),
    LogTransformer(),
    LogCpTransformer(),
    ArcsinTransformer(),
    ArcSinhTransformer(),
    PowerTransformer(),
    ReciprocalTransformer(),
    YeoJohnsonTransformer(),
]

checks_with_negative_values = [
    "check_readonly_memmap_input",
    "check_fit_score_takes_y",
    "check_dont_overwrite_parameters",
    "check_estimators_nan_inf",
    "check_f_contiguous_array_estimator",
    "check_fit2d_1feature",
    "check_fit2d_1sample",
    "check_dict_unchanged",
    "check_fit_check_is_fitted",
    "check_n_features_in",
    "check_positive_only_tag_during_fit",
    "check_methods_subset_invariance",
]
estimators_not_supporting_negative_values = [
    "BoxCoxTransformer",
    "LogTransformer",
    "ArcsinTransformer",
]
extra_failing_checks = {
    estimator_name: dict.fromkeys(
        checks_with_negative_values,
        "this checks passes a negative value which is not supported by "
        "the transformer",
    )
    for estimator_name in estimators_not_supporting_negative_values
}


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    expected_failed_checks = estimator._more_tags()["_xfail_checks"]
    expected_failed_checks.update(
        extra_failing_checks.get(estimator.__class__.__name__, {})
    )
    return check_estimator(
        estimator=estimator,
        expected_failed_checks=expected_failed_checks,
    )


@pytest.mark.parametrize("estimator", _estimators[4:])
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_in_pipeline_with_set_output_pandas(transformer):
    X = pd.DataFrame(
        {"feature_1": [0.1, 0.2, 0.3, 0.4, 0.5], "feature_2": [0.6, 0.7, 0.8, 0.9, 0.1]}
    )
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)


@pytest.mark.parametrize("estimator", _estimators)
def test_raises_non_fitted_error_when_error_during_fit(estimator):
    name = estimator.__class__.__name__

    if name == "BoxCoxTransformer":
        # non-positive values: boxcox itself raises after variables_ selection.
        X = pd.DataFrame({"num1": [-1.0, 2.0, 3.0, 4.0, 5.0]})
    elif name == "LogTransformer":
        # default C=0: zero/negative values raise after variables_ selection.
        X = pd.DataFrame({"num1": [-1.0, 2.0, 3.0, 4.0, 5.0]})
    elif name == "ArcsinTransformer":
        # values outside 0-1: raises after variables_ selection.
        X = pd.DataFrame({"num1": [1.1, 2.0, 3.0, 4.0, 5.0]})
    elif name == "ReciprocalTransformer":
        # zero values: raises after variables_ selection.
        X = pd.DataFrame({"num1": [0.0, 2.0, 3.0, 4.0, 5.0]})
    else:
        # LogCpTransformer (C="auto" never fails on the values themselves),
        # ArcSinhTransformer, PowerTransformer, YeoJohnsonTransformer: none of
        # these validate values beyond being numerical, so there is no
        # reachable failure point once variables_ would be selected. Fail at
        # variable selection instead.
        X = pd.DataFrame({"cat1": ["a", "b", "c", "a", "b"]})

    check_raises_non_fitted_error_when_fit_fails(estimator, X)
