from sklearn.impute import SimpleImputer
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.wrappers import SklearnTransformerWrapper


def test_sklearn_transformer_wrapper():
    check_estimator(SklearnTransformerWrapper(transformer=SimpleImputer()))
