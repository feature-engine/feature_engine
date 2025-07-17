"""
File intended to help understand check_estimator tests for Feature-engine's
selection module. It is not run as part of the battery of acceptance tests.
Works from sklearn > 1.6.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from feature_engine.selection import (
    MRMR,
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
    DropFeatures,
    DropHighPSIFeatures,
    ProbeFeatureSelection,
    RecursiveFeatureAddition,
    RecursiveFeatureElimination,
    SelectByInformationValue,
    SelectByShuffling,
    SelectBySingleFeaturePerformance,
    SelectByTargetMeanPerformance,
    SmartCorrelatedSelection,
)

_logreg = LogisticRegression(C=0.0001, max_iter=2, random_state=1)

df = DropFeatures(features_to_drop=["x0"])
dcf = DropConstantFeatures(missing_values="ignore")
ddf = DropDuplicateFeatures()
dcf = DropCorrelatedFeatures()
dpsi = DropHighPSIFeatures(bins=5)
sms = SmartCorrelatedSelection()
sbs = SelectByShuffling(estimator=_logreg, scoring="accuracy")
sbtm = SelectByTargetMeanPerformance(bins=3, regression=True, scoring="r2")
sbsfp = SelectBySingleFeaturePerformance(estimator=_logreg, scoring="accuracy")
rfa = RecursiveFeatureAddition(estimator=_logreg, scoring="accuracy")
rfe = RecursiveFeatureElimination(estimator=_logreg, scoring="accuracy", threshold=-100)
sbiv = SelectByInformationValue(bins=2)
pfs = ProbeFeatureSelection(estimator=_logreg, scoring="accuracy")
mrmr = MRMR(regression=False)

EXPECTED_FAILED_CHECKS = {
    "DropFeatures": df._more_tags()["_xfail_checks"],
    "DropConstantFeatures": dcf._more_tags()["_xfail_checks"],
    "DropDuplicateFeatures": ddf._more_tags()["_xfail_checks"],
    "DropCorrelatedFeatures": dcf._more_tags()["_xfail_checks"],
    "DropHighPSIFeatures": dpsi._more_tags()["_xfail_checks"],
    "SmartCorrelatedSelection": sms._more_tags()["_xfail_checks"],
    "SelectByShuffling": sbs._more_tags()["_xfail_checks"],
    "SelectByTargetMeanPerformance": sbtm._more_tags()["_xfail_checks"],
    "SelectBySingleFeaturePerformance": sbsfp._more_tags()["_xfail_checks"],
    "RecursiveFeatureAddition": rfa._more_tags()["_xfail_checks"],
    "RecursiveFeatureElimination": rfe._more_tags()["_xfail_checks"],
    "SelectByInformationValue": sbiv._more_tags()["_xfail_checks"],
    "ProbeFeatureSelection": pfs._more_tags()["_xfail_checks"],
    "MRMR": mrmr._more_tags()["_xfail_checks"],
}


# encoding
@parametrize_with_checks(
    estimators=[
        df,
        dcf,
        ddf,
        dcf,
        dpsi,
        sms,
        sbs,
        sbtm,
        sbsfp,
        rfa,
        rfe,
        sbiv,
        pfs,
        mrmr,
    ],
    expected_failed_checks=lambda est: EXPECTED_FAILED_CHECKS.get(
        est.__class__.__name__, {}
    ),
)
def test_sklearn_compatible_creator(estimator, check):
    check(estimator)
