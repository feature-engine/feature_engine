from sklearn.utils.estimator_checks import parametrize_with_checks

from feature_engine.creation import (
    CyclicalFeatures,
    DecisionTreeFeatures,
    MathFeatures,
    RelativeFeatures,
)

dtf = DecisionTreeFeatures(regression=False)
cf = CyclicalFeatures()
mf = MathFeatures(variables=["x0", "x1"], func="mean", missing_values="ignore")
rf = RelativeFeatures(
    variables=["x0", "x1"],
    reference=["x0"],
    func=["add"],
    missing_values="ignore",
)

EXPECTED_FAILED_CHECKS = {
    "DecisionTreeFeatures": dtf._more_tags()['_xfail_checks'],
    "CyclicalFeatures": cf._more_tags()['_xfail_checks'],
    "MathFeatures": mf._more_tags()['_xfail_checks'],
    "RelativeFeatures": rf._more_tags()['_xfail_checks'],
}
# creation
@parametrize_with_checks(
    estimators = [dtf, cf, mf, rf],
    expected_failed_checks=lambda est: EXPECTED_FAILED_CHECKS.get(est.__class__.__name__, {})
)
def test_sklearn_compatible_creator(estimator, check):
    check(estimator)
