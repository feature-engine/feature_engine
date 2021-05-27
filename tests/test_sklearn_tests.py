from sklearn.utils.estimator_checks import parametrize_with_checks

from feature_engine.imputation import (
    MeanMedianImputer,
    ArbitraryNumberImputer,
    CategoricalImputer,
    EndTailImputer,
    AddMissingIndicator,
    RandomSampleImputer,
    DropMissingData,
)
from feature_engine.encoding import (
    CountFrequencyEncoder,
    DecisionTreeEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
    WoEEncoder,
    PRatioEncoder,
)
from feature_engine.transformation import (
    BoxCoxTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)
from feature_engine.wrappers import SklearnTransformerWrapper


@parametrize_with_checks([
    MeanMedianImputer(),
    ArbitraryNumberImputer(),
    CategoricalImputer(),
    EndTailImputer(),
    AddMissingIndicator(),
    RandomSampleImputer(),
    DropMissingData(),
])
def test_sklearn_compatible_imputer(estimator, check):
    check(estimator)


@parametrize_with_checks([
    CountFrequencyEncoder(),
    DecisionTreeEncoder(),
    MeanEncoder(),
    OneHotEncoder(),
    OrdinalEncoder(),
    RareLabelEncoder(),
    WoEEncoder(),
    PRatioEncoder(),
])
def test_sklearn_compatible_encoder(estimator, check):
    check(estimator)


@parametrize_with_checks([BoxCoxTransformer(), LogTransformer(), PowerTransformer(),
                          ReciprocalTransformer(), YeoJohnsonTransformer(),
                          ])
def test_sklearn_compatible_transformer(estimator, check):
    check(estimator)

