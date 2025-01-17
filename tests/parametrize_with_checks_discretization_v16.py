"""
File intended to help understand check_estimator tests for Feature-engine's
discretization module. It is not run as part of the battery of acceptance tests.
Works from sklearn > 1.6.
"""

import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks

from feature_engine.discretisation import (
    ArbitraryDiscretiser,
    DecisionTreeDiscretiser,
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    GeometricWidthDiscretiser,
)

dtd = DecisionTreeDiscretiser(regression=False)
efd = EqualFrequencyDiscretiser()
ewd = EqualWidthDiscretiser()
ad = ArbitraryDiscretiser(binning_dict={"x0": [-np.inf, 0, np.inf]})
gd = GeometricWidthDiscretiser()

EXPECTED_FAILED_CHECKS = {
    "DecisionTreeDiscretiser": dtd._more_tags()["_xfail_checks"],
    "EqualFrequencyDiscretiser": efd._more_tags()["_xfail_checks"],
    "EqualWidthDiscretiser": ewd._more_tags()["_xfail_checks"],
    "ArbitraryDiscretiser": ad._more_tags()["_xfail_checks"],
    "GeometricWidthDiscretiser": gd._more_tags()["_xfail_checks"],
}


# discretization
@parametrize_with_checks(
    estimators=[dtd, efd, ewd, ad, gd],
    expected_failed_checks=lambda est: EXPECTED_FAILED_CHECKS.get(
        est.__class__.__name__, {}
    ),
)
def test_sklearn_compatible_creator(estimator, check):
    check(estimator)
