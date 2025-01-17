"""
File intended to help understand check_estimator tests for Feature-engine's
encoding module. It is not run as part of the battery of acceptance tests.
Works from sklearn > 1.6.
"""

from sklearn.utils.estimator_checks import parametrize_with_checks

from feature_engine.encoding import (
    CountFrequencyEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
    StringSimilarityEncoder,
    WoEEncoder,
)
from feature_engine.tags import _return_tags

ce = CountFrequencyEncoder(ignore_format=True)
me = MeanEncoder(ignore_format=True)
ohe = OneHotEncoder(ignore_format=True)
oe = OrdinalEncoder(ignore_format=True)
re = RareLabelEncoder(
    tol=0.00000000001,
    n_categories=100000000000,
    replace_with=10,
    ignore_format=True,
)
woe = WoEEncoder(ignore_format=True)
sse = StringSimilarityEncoder(ignore_format=True)

FAILED_CHECKS = _return_tags()["_xfail_checks"]
FAILED_CHECKS.update({"check_estimators_nan_inf": "transformer allows NA"})

EXPECTED_FAILED_CHECKS = {
    "CountFrequencyEncoder": FAILED_CHECKS,
    "MeanEncoder": FAILED_CHECKS,
    "OneHotEncoder": FAILED_CHECKS,
    "OrdinalEncoder": FAILED_CHECKS,
    "RareLabelEncoder": FAILED_CHECKS,
    "StringSimilarityEncoder": FAILED_CHECKS,
}


# encoding
@parametrize_with_checks(
    estimators=[ce, me, ohe, oe, re, woe, sse],
    expected_failed_checks=lambda est: EXPECTED_FAILED_CHECKS.get(
        est.__class__.__name__, {}
    ),
)
def test_sklearn_compatible_creator(estimator, check):
    check(estimator)
