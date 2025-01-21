import pandas as pd
import pytest
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine.transformation import (
    ArcsinTransformer,
    BoxCoxTransformer,
    LogCpTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

_estimators = [
    BoxCoxTransformer(),
    LogTransformer(),
    LogCpTransformer(),
    ArcsinTransformer(),
    PowerTransformer(),
    ReciprocalTransformer(),
    YeoJohnsonTransformer(),
]

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

if sklearn_version < parse_version("1.6"):

    @pytest.mark.parametrize("estimator", _estimators)
    def test_check_estimator_from_sklearn(estimator):
        return check_estimator(estimator)

else:
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
        estimator_name: {
            check: "this checks passes a negative value which is not supported by the "
            "transformer"
            for check in checks_with_negative_values
        }
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
