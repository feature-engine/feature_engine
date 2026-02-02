"""
File intended to help understand check_estimator tests for Feature-engine's
outliers module. It is not run as part of the battery of acceptance tests.
Works from sklearn > 1.6.
"""

from sklearn.utils.estimator_checks import parametrize_with_checks

from feature_engine.outliers import ArbitraryOutlierCapper, OutlierTrimmer, Winsorizer
from feature_engine.tags import _return_tags

aoc = ArbitraryOutlierCapper(max_capping_dict={"x0": 10})
ot = OutlierTrimmer()
wz = Winsorizer()

FAILED_CHECKS = _return_tags()["_xfail_checks"]
FAILED_CHECKS_AOC = _return_tags()["_xfail_checks"]

msg1 = "transformers raise errors when data variation is low, thus this check fails"

msg2 = "transformer has 1 mandatory parameter"

FAILED_CHECKS.update({"check_fit2d_1sample": msg1})
FAILED_CHECKS_AOC.update(
    {
        "check_fit2d_1sample": msg1,
        "check_parameters_default_constructible": msg2,
    }
)

EXPECTED_FAILED_CHECKS = {
    "ArbitraryOutlierCapper": FAILED_CHECKS_AOC,
    "OutlierTrimmer": FAILED_CHECKS,
    "Winsorizer": FAILED_CHECKS,
}


# encoding
@parametrize_with_checks(
    estimators=[aoc, ot, wz],
    expected_failed_checks=lambda est: EXPECTED_FAILED_CHECKS.get(
        est.__class__.__name__, {}
    ),
)
def test_sklearn_compatible_creator(estimator, check):
    check(estimator)
