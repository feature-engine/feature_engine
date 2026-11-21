import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.imputation import (
    MissingIndicator,
    ArbitraryImputer,
    CategoricalImputer,
    DropMissingData,
    EndTailImputer,
    MeanImputer,
    RandomSampleImputer,
)
from tests.backend_helpers import frame_to_dict
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator
from tests.estimator_checks.non_fitted_error_checks import (
    check_raises_non_fitted_error_when_fit_fails,
)
from tests.estimator_checks.sklearn_check_wrapper import wrap_for_check_estimator

_estimators = [
    MeanImputer(),
    ArbitraryImputer(),
    CategoricalImputer(fill_value=0, ignore_format=True),
    EndTailImputer(),
    MissingIndicator(),
    RandomSampleImputer(),
    DropMissingData(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(
        estimator=wrap_for_check_estimator(estimator),
        expected_failed_checks=estimator._more_tags()["_xfail_checks"],
    )


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    if estimator.__class__.__name__ == "CategoricalImputer":
        estimator.set_params(ignore_format=False)
    if estimator.__class__.__name__ in ["DropMissingData", "MissingIndicator"]:
        estimator.set_params(missing_only=False)
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_in_pipeline_with_set_output_pandas(transformer):
    if transformer.__class__.__name__ == "CategoricalImputer":
        transformer.set_params(ignore_format=True)
    if transformer.__class__.__name__ in ["DropMissingData", "MissingIndicator"]:
        transformer.set_params(missing_only=False)

    X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [6, 7, 8, 9, 10],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="pandas")

    Xtt = transformer.fit_transform(X)
    Xtp = pipe.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xtt, Xtp)


@pytest.mark.parametrize("estimator", _estimators)
def test_raises_non_fitted_error_when_error_during_fit(estimator):
    if estimator.__class__.__name__ in ["MeanImputer", "EndTailImputer"]:
        # no numerical variables in the df: fails at variable selection.
        X = pd.DataFrame({"cat1": ["a", "b", "c", "a", "b"]})
    elif estimator.__class__.__name__ == "ArbitraryImputer":
        X = pd.DataFrame({"cat1": ["a", "b", "c", "a", "b"]})
    elif estimator.__class__.__name__ == "RandomSampleImputer":
        # invalid random_state: fails after variables_/X_ would have been set.
        estimator = RandomSampleImputer(seed="observation", random_state="not_a_col")
        X = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    else:
        # CategoricalImputer, AddMissingIndicator, DropMissingData: no
        # reachable failure point once variables are selected, so fail at
        # input validation instead.
        X = pd.DataFrame()

    check_raises_non_fitted_error_when_fit_fails(estimator, X)


_REORDER_DATA = {
    "num_1": [1.0, None, 3.0, 4.0, 5.0],
    "num_2": [10.0, 20.0, None, 40.0, 50.0],
    "cat": ["a", "b", None, "a", "a"],
}


@pytest.mark.parametrize(
    "transformer",
    [
        MeanImputer(),
        ArbitraryImputer(),
        EndTailImputer(),
        CategoricalImputer(),
        CategoricalImputer(imputation_method="frequent"),
        MissingIndicator(),
        RandomSampleImputer(random_state=0),
        RandomSampleImputer(seed="observation", random_state="num_1"),
        DropMissingData(),
    ],
)
def test_transform_reorders_columns_as_in_train_set(make_df, transformer):
    X = make_df(_REORDER_DATA)
    expected = transformer.fit(X).transform(X)

    reordered = ["cat", "num_2", "num_1"]
    X_reordered = make_df({var: _REORDER_DATA[var] for var in reordered})
    Xt = transformer.transform(X_reordered)

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == list(expected.columns)
    assert frame_to_dict(Xt) == frame_to_dict(expected)
