from sklearn.utils.estimator_checks import check_estimator
from feature_engine.wrappers import SklearnTransformerWrapper


def test_sklearn_trainsformer_wrapper():
    check_estimator(SklearnTransformerWrapper())
