"""
This file is only intended to help understand check_estimator tests on Feature-engine
transformers. It is not run as part of the battery of acceptance tests.
"""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from feature_engine.encoding import (
    CountFrequencyEncoder,
    DecisionTreeEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PRatioEncoder,
    RareLabelEncoder,
    WoEEncoder,
)
from feature_engine.imputation import (
    AddMissingIndicator,
    ArbitraryNumberImputer,
    CategoricalImputer,
    DropMissingData,
    EndTailImputer,
    MeanMedianImputer,
    RandomSampleImputer,
)
from feature_engine.outliers import ArbitraryOutlierCapper, OutlierTrimmer, Winsorizer
from feature_engine.selection import (
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
    DropFeatures,
    DropHighPSIFeatures,
    RecursiveFeatureAddition,
    RecursiveFeatureElimination,
    SelectByShuffling,
    SelectBySingleFeaturePerformance,
    SelectByTargetMeanPerformance,
    SmartCorrelatedSelection,
)
from feature_engine.transformation import (
    BoxCoxTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)
from feature_engine.wrappers import SklearnTransformerWrapper


# imputation
@parametrize_with_checks(
    [
        MeanMedianImputer(),
        ArbitraryNumberImputer(),
        CategoricalImputer(fill_value=0, ignore_format=True),
        EndTailImputer(),
        AddMissingIndicator(),
        RandomSampleImputer(),
        DropMissingData(),
    ]
)
def test_sklearn_compatible_imputer(estimator, check):
    check(estimator)


# encoding
@parametrize_with_checks(
    [
        CountFrequencyEncoder(ignore_format=True),
        DecisionTreeEncoder(regression=False, ignore_format=True),
        MeanEncoder(ignore_format=True),
        OneHotEncoder(ignore_format=True),
        OrdinalEncoder(ignore_format=True),
        RareLabelEncoder(
            tol=0.00000000001,
            n_categories=100000000000,
            replace_with=10,
            ignore_format=True,
        ),
        WoEEncoder(ignore_format=True),
        PRatioEncoder(ignore_format=True),
    ]
)
def test_sklearn_compatible_encoder(estimator, check):
    check(estimator)


# outliers
@parametrize_with_checks(
    [
        ArbitraryOutlierCapper(max_capping_dict={"0": 10}),
        OutlierTrimmer(),
        Winsorizer(),
    ]
)
def test_sklearn_compatible_outliers(estimator, check):
    check(estimator)


# transformers
@parametrize_with_checks(
    [
        BoxCoxTransformer(),
        LogTransformer(),
        PowerTransformer(),
        ReciprocalTransformer(),
        YeoJohnsonTransformer(),
    ]
)
def test_sklearn_compatible_transformer(estimator, check):
    check(estimator)


# selectors
@parametrize_with_checks(
    [
        DropFeatures(features_to_drop=["0"]),
        DropConstantFeatures(missing_values="ignore"),
        DropDuplicateFeatures(),
        DropCorrelatedFeatures(),
        SmartCorrelatedSelection(),
        DropHighPSIFeatures(bins=5),
        SelectByShuffling(
            LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
        ),
        SelectBySingleFeaturePerformance(
            LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
        ),
        RecursiveFeatureAddition(
            LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
        ),
        RecursiveFeatureElimination(
            LogisticRegression(max_iter=2, random_state=1), scoring="accuracy"
        ),
        SelectByTargetMeanPerformance(scoring="r2_score", bins=3),
    ]
)
def test_sklearn_compatible_selectors(estimator, check):
    check(estimator)


# wrappers
@parametrize_with_checks([SklearnTransformerWrapper(SimpleImputer())])
def test_sklearn_compatible_wrapper(estimator, check):
    check(estimator)
