"""
Based on imbalanced learn pipeline's tests:
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tests/test_pipeline.py

And sklearn's pipeline's tests:
https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/tests/test_pipeline.py
"""

import itertools
import re
import shutil
from tempfile import mkdtemp

import numpy as np
import pytest
from joblib import Memory
from pytest import raises
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.tests.test_pipeline import (
    NoFit,
    NoTrans,
    NoInvTransf,
    Transf,
    TransfFitParams,
    Mult,
    FitParamT,
    DummyTransf,
    DummyEstimatorParams,
)

from feature_engine.pipeline import Pipeline, make_pipeline

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)

R_TOL = 1e-4


def test_pipeline_invalid_parameters():
    # Test the various init parameters of the pipeline in fit
    # method
    pipeline = Pipeline([(1, 1)])
    with pytest.raises(TypeError):
        pipeline.fit([[1]], [1])

    # Check that we can't fit pipelines with objects without fit
    # method
    msg = (
        "Last step of Pipeline should implement fit "
        "or be the string 'passthrough'"
        ".*NoFit.*"
    )
    pipeline = Pipeline([("clf", NoFit())])
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])

    # Smoke test with only an estimator
    clf = NoTrans()
    pipe = Pipeline([("svc", clf)])
    assert pipe.get_params(deep=True) == dict(
        svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False)
    )

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    repr(pipe)

    # Test with two objects
    clf = SVC()
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([("anova", filter1), ("svc", clf)])

    # Check that estimators are not cloned on pipeline construction
    assert pipe.named_steps["anova"] is filter1
    assert pipe.named_steps["svc"] is clf

    # Check that we can't fit with non-transformers on the way
    # Note that NoTrans implements fit, but not transform
    msg = "All intermediate steps should be transformers.*\\bNoTrans\\b.*"
    pipeline = Pipeline([("t", NoTrans()), ("svc", clf)])
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    repr(pipe)

    # Check that params are not set when naming them wrong
    msg = re.escape(
        "Invalid parameter 'C' for estimator SelectKBest(). Valid parameters are: ['k',"
        " 'score_func']."
    )
    with pytest.raises(ValueError, match=msg):
        pipe.set_params(anova__C=0.1)

    # Test clone
    pipe2 = clone(pipe)
    assert pipe.named_steps["svc"] is not pipe2.named_steps["svc"]

    # Check that apart from estimators, the parameters are the same
    params = pipe.get_params(deep=True)
    params2 = pipe2.get_params(deep=True)

    for x in pipe.get_params(deep=False):
        params.pop(x)

    for x in pipe2.get_params(deep=False):
        params2.pop(x)

    # Remove estimators that where copied
    params.pop("svc")
    params.pop("anova")
    params2.pop("svc")
    params2.pop("anova")
    assert params == params2


def test_pipeline_init_tuple():
    # Pipeline accepts steps as tuple
    X = np.array([[1, 2]])
    pipe = Pipeline((("transf", Transf()), ("clf", FitParamT())))
    pipe.fit(X, y=None)
    pipe.score(X)
    pipe.set_params(transf="passthrough")
    pipe.fit(X, y=None)
    pipe.score(X)


def test_pipeline_init():
    # Test the various init parameters of the pipeline.
    with raises(TypeError):
        Pipeline()
    # Check that we can't instantiate pipelines with objects without fit
    # method
    X, y = load_iris(return_X_y=True)
    error_regex = (
        "Last step of Pipeline should implement fit or be the string 'passthrough'"
    )
    with raises(TypeError, match=error_regex):
        model = Pipeline([("clf", NoFit())])
        model.fit(X, y)
    # Smoke test with only an estimator
    clf = NoTrans()
    pipe = Pipeline([("svc", clf)])
    expected = dict(svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False))
    assert pipe.get_params(deep=True) == expected

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    repr(pipe)

    # Test with two objects
    clf = SVC(gamma="scale")
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([("anova", filter1), ("svc", clf)])

    # Check that we can't instantiate with non-transformers on the way
    # Note that NoTrans implements fit, but not transform
    error_regex = "All intermediate steps should be transformers.*\\bNoTrans\\b.*"
    with raises(TypeError, match=error_regex):
        model = Pipeline([("t", NoTrans()), ("svc", clf)])
        model.fit(X, y)

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    repr(pipe)

    # Check that params are not set when naming them wrong
    with raises(ValueError):
        pipe.set_params(anova__C=0.1)

    # Test clone
    pipe2 = clone(pipe)
    assert pipe.named_steps["svc"] is not pipe2.named_steps["svc"]

    # Check that apart from estimators, the parameters are the same
    params = pipe.get_params(deep=True)
    params2 = pipe2.get_params(deep=True)

    for x in pipe.get_params(deep=False):
        params.pop(x)

    for x in pipe2.get_params(deep=False):
        params2.pop(x)

    # Remove estimators that where copied
    params.pop("svc")
    params.pop("anova")
    params2.pop("svc")
    params2.pop("anova")
    assert params == params2


def test_pipeline_methods_anova():
    test_pipeline_methods_anova
    # Test the various methods of the pipeline (anova).
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Test with Anova + LogisticRegression
    clf = LogisticRegression(solver="lbfgs")
    filter1 = SelectKBest(f_classif, k=2)
    pipe = Pipeline([("anova", filter1), ("logistic", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_fit_params():
    # Test that the pipeline can take fit parameters
    pipe = Pipeline([("transf", Transf()), ("clf", FitParamT())])
    pipe.fit(X=None, y=None, clf__should_succeed=True)
    # classifier should return True
    assert pipe.predict(None)
    # and transformer params should not be changed
    assert pipe.named_steps["transf"].a is None
    assert pipe.named_steps["transf"].b is None
    # invalid parameters should raise an error message
    with raises(TypeError, match="unexpected keyword argument"):
        pipe.fit(None, None, clf__bad=True)


def test_pipeline_sample_weight_supported():
    # Pipeline should pass sample_weight
    X = np.array([[1, 2]])
    pipe = Pipeline([("transf", Transf()), ("clf", FitParamT())])
    pipe.fit(X, y=None)
    assert pipe.score(X) == 3
    assert pipe.score(X, y=None) == 3
    assert pipe.score(X, y=None, sample_weight=None) == 3
    assert pipe.score(X, sample_weight=np.array([2, 3])) == 8


def test_pipeline_sample_weight_unsupported():
    # When sample_weight is None it shouldn't be passed
    X = np.array([[1, 2]])
    pipe = Pipeline([("transf", Transf()), ("clf", Mult())])
    pipe.fit(X, y=None)
    assert pipe.score(X) == 3
    assert pipe.score(X, sample_weight=None) == 3
    with raises(TypeError, match="unexpected keyword argument"):
        pipe.score(X, sample_weight=np.array([2, 3]))


def test_pipeline_raise_set_params_error():
    # Test pipeline raises set params error message for nested models.
    pipe = Pipeline([("cls", LinearRegression())])
    with raises(ValueError, match="Invalid parameter"):
        pipe.set_params(fake="nope")

    # nested model check
    with raises(ValueError, match="Invalid parameter"):
        pipe.set_params(fake__estimator="nope")


def test_pipeline_methods_pca_svm():
    # Test the various methods of the pipeline (pca + svm).
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Test with PCA + SVC
    clf = SVC(gamma="scale", probability=True, random_state=0)
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    pipe = Pipeline([("pca", pca), ("svc", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_methods_preprocessing_svm():
    # Test the various methods of the pipeline (preprocessing + svm).
    iris = load_iris()
    X = iris.data
    y = iris.target
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    scaler = StandardScaler()
    pca = PCA(n_components=2, svd_solver="randomized", whiten=True)
    clf = SVC(
        gamma="scale",
        probability=True,
        random_state=0,
        decision_function_shape="ovr",
    )

    for preprocessing in [scaler, pca]:
        pipe = Pipeline([("preprocess", preprocessing), ("svc", clf)])
        pipe.fit(X, y)

        # check shapes of various prediction functions
        predict = pipe.predict(X)
        assert predict.shape == (n_samples,)

        proba = pipe.predict_proba(X)
        assert proba.shape == (n_samples, n_classes)

        log_proba = pipe.predict_log_proba(X)
        assert log_proba.shape == (n_samples, n_classes)

        decision_function = pipe.decision_function(X)
        assert decision_function.shape == (n_samples, n_classes)

        pipe.score(X, y)


def test_fit_predict_on_pipeline():
    # test that the fit_predict method is implemented on a pipeline
    # test that the fit_predict on pipeline yields same results as applying
    # transform and clustering steps separately
    iris = load_iris()
    scaler = StandardScaler()
    km = KMeans(random_state=0, n_init=10)
    # As pipeline doesn't clone estimators on construction,
    # it must have its own estimators
    scaler_for_pipeline = StandardScaler()
    km_for_pipeline = KMeans(random_state=0, n_init=10)

    # first compute the transform and clustering step separately
    scaled = scaler.fit_transform(iris.data)
    separate_pred = km.fit_predict(scaled)

    # use a pipeline to do the transform and clustering in one step
    pipe = Pipeline([("scaler", scaler_for_pipeline), ("Kmeans", km_for_pipeline)])
    pipeline_pred = pipe.fit_predict(iris.data)

    assert_array_almost_equal(pipeline_pred, separate_pred)


def test_fit_predict_on_pipeline_without_fit_predict():
    # tests that a pipeline does not have fit_predict method when final
    # step of pipeline does not have fit_predict defined
    scaler = StandardScaler()
    pca = PCA(svd_solver="full")
    pipe = Pipeline([("scaler", scaler), ("pca", pca)])

    outer_msg = "'Pipeline' has no attribute 'fit_predict'"
    inner_msg = "'PCA' object has no attribute 'fit_predict'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        getattr(pipe, "fit_predict")
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)


def test_fit_predict_with_intermediate_fit_params():
    # tests that Pipeline passes fit_params to intermediate steps
    # when fit_predict is invoked
    pipe = Pipeline([("transf", TransfFitParams()), ("clf", FitParamT())])
    pipe.fit_predict(
        X=None, y=None, transf__should_get_this=True, clf__should_succeed=True
    )
    assert pipe.named_steps["transf"].fit_params["should_get_this"]
    assert pipe.named_steps["clf"].successful
    assert "should_succeed" not in pipe.named_steps["transf"].fit_params


def test_pipeline_transform():
    # Test whether pipeline works with a transformer at the end.
    # Also test pipeline.transform and pipeline.inverse_transform
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=2, svd_solver="full")
    pipeline = Pipeline([("pca", pca)])

    # test transform and fit_transform:
    X_trans = pipeline.fit(X).transform(X)
    X_trans2 = pipeline.fit_transform(X)
    X_trans3 = pca.fit_transform(X)
    assert_array_almost_equal(X_trans, X_trans2)
    assert_array_almost_equal(X_trans, X_trans3)

    X_back = pipeline.inverse_transform(X_trans)
    X_back2 = pca.inverse_transform(X_trans)
    assert_array_almost_equal(X_back, X_back2)


def test_pipeline_fit_transform():
    # Test whether pipeline works with a transformer missing fit_transform
    iris = load_iris()
    X = iris.data
    y = iris.target
    transf = Transf()
    pipeline = Pipeline([("mock", transf)])

    # test fit_transform:
    X_trans = pipeline.fit_transform(X, y)
    X_trans2 = transf.fit(X, y).transform(X)
    assert_array_almost_equal(X_trans, X_trans2)


def test_set_pipeline_steps():
    transf1 = Transf()
    transf2 = Transf()
    pipeline = Pipeline([("mock", transf1)])
    assert pipeline.named_steps["mock"] is transf1

    # Directly setting attr
    pipeline.steps = [("mock2", transf2)]
    assert "mock" not in pipeline.named_steps
    assert pipeline.named_steps["mock2"] is transf2
    assert [("mock2", transf2)] == pipeline.steps

    # Using set_params
    pipeline.set_params(steps=[("mock", transf1)])
    assert [("mock", transf1)] == pipeline.steps

    # Using set_params to replace single step
    pipeline.set_params(mock=transf2)
    assert [("mock", transf2)] == pipeline.steps

    # With invalid data
    pipeline.set_params(steps=[("junk", ())])
    with raises(TypeError):
        pipeline.fit([[1]], [1])
    with raises(AttributeError):
        pipeline.fit_transform([[1]], [1])


@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_pipeline_correctly_adjusts_steps(passthrough):
    X = np.array([[1]])
    y = np.array([1])
    mult2 = Mult(mult=2)
    mult3 = Mult(mult=3)
    mult5 = Mult(mult=5)
    pipeline = Pipeline(
        [("m2", mult2), ("bad", passthrough), ("m3", mult3), ("m5", mult5)]
    )
    pipeline.fit(X, y)
    expected_names = ["m2", "bad", "m3", "m5"]
    actual_names = [name for name, _ in pipeline.steps]
    assert expected_names == actual_names


@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_set_pipeline_step_passthrough(passthrough):
    # Test setting Pipeline steps to None
    X = np.array([[1]])
    y = np.array([1])
    mult2 = Mult(mult=2)
    mult3 = Mult(mult=3)
    mult5 = Mult(mult=5)

    def make():
        return Pipeline([("m2", mult2), ("m3", mult3), ("last", mult5)])

    pipeline = make()

    exp = 2 * 3 * 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    pipeline.set_params(m3=passthrough)
    exp = 2 * 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    expected_params = {
        "steps": pipeline.steps,
        "m2": mult2,
        "m3": passthrough,
        "last": mult5,
        "memory": None,
        "m2__mult": 2,
        "last__mult": 5,
        "verbose": False,
    }
    assert pipeline.get_params(deep=True) == expected_params

    pipeline.set_params(m2=passthrough)
    exp = 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    # for other methods, ensure no AttributeErrors on None:
    other_methods = [
        "predict_proba",
        "predict_log_proba",
        "decision_function",
        "transform",
        "score",
    ]
    for method in other_methods:
        getattr(pipeline, method)(X)

    pipeline.set_params(m2=mult2)
    exp = 2 * 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    pipeline = make()
    pipeline.set_params(last=passthrough)
    # mult2 and mult3 are active
    exp = 6
    pipeline.fit(X, y)
    pipeline.transform(X)
    assert_array_equal([[exp]], pipeline.fit(X, y).transform(X))
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    with raises(AttributeError, match="has no attribute 'predict'"):
        getattr(pipeline, "predict")

    # Check 'passthrough' step at construction time
    exp = 2 * 5
    pipeline = Pipeline([("m2", mult2), ("m3", passthrough), ("last", mult5)])
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))


def test_pipeline_ducktyping():
    pipeline = make_pipeline(Mult(5))
    pipeline.predict
    pipeline.transform
    pipeline.inverse_transform

    pipeline = make_pipeline(Transf())
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    pipeline.inverse_transform

    pipeline = make_pipeline("passthrough")
    assert pipeline.steps[0] == ("passthrough", "passthrough")
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    pipeline.inverse_transform

    pipeline = make_pipeline(Transf(), NoInvTransf())
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    assert not hasattr(pipeline, "inverse_transform")

    pipeline = make_pipeline(NoInvTransf(), Transf())
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    assert not hasattr(pipeline, "inverse_transform")


def test_make_pipeline():
    t1 = Transf()
    t2 = Transf()
    pipe = make_pipeline(t1, t2)
    assert isinstance(pipe, Pipeline)
    assert pipe.steps[0][0] == "transf-1"
    assert pipe.steps[1][0] == "transf-2"

    pipe = make_pipeline(t1, t2, FitParamT())
    assert isinstance(pipe, Pipeline)
    assert pipe.steps[0][0] == "transf-1"
    assert pipe.steps[1][0] == "transf-2"
    assert pipe.steps[2][0] == "fitparamt"


def test_classes_property():
    iris = load_iris()
    X = iris.data
    y = iris.target

    reg = make_pipeline(SelectKBest(k=1), LinearRegression())
    reg.fit(X, y)
    with raises(AttributeError):
        getattr(reg, "classes_")

    clf = make_pipeline(
        SelectKBest(k=1),
        LogisticRegression(solver="lbfgs", random_state=0),
    )
    with raises(AttributeError):
        getattr(clf, "classes_")
    clf.fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))


def test_pipeline_memory_transformer():
    iris = load_iris()
    X = iris.data
    y = iris.target
    cachedir = mkdtemp()
    try:
        memory = Memory(cachedir, verbose=10)
        # Test with Transformer + SVC
        clf = SVC(gamma="scale", probability=True, random_state=0)
        transf = DummyTransf()
        pipe = Pipeline([("transf", clone(transf)), ("svc", clf)])
        cached_pipe = Pipeline([("transf", transf), ("svc", clf)], memory=memory)

        # Memoize the transformer at the first fit
        cached_pipe.fit(X, y)
        pipe.fit(X, y)
        # Get the time stamp of the tranformer in the cached pipeline
        expected_ts = cached_pipe.named_steps["transf"].timestamp_
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X), cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe.named_steps["transf"].means_,
        )
        assert not hasattr(transf, "means_")
        # Check that we are reading the cache while fitting
        # a second time
        cached_pipe.fit(X, y)
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X), cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe.named_steps["transf"].means_,
        )
        assert cached_pipe.named_steps["transf"].timestamp_ == expected_ts
        # Create a new pipeline with cloned estimators
        # Check that even changing the name step does not affect the cache hit
        clf_2 = SVC(gamma="scale", probability=True, random_state=0)
        transf_2 = DummyTransf()
        cached_pipe_2 = Pipeline(
            [("transf_2", transf_2), ("svc", clf_2)], memory=memory
        )
        cached_pipe_2.fit(X, y)

        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe_2.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe_2.predict_proba(X))
        assert_array_equal(
            pipe.predict_log_proba(X), cached_pipe_2.predict_log_proba(X)
        )
        assert_array_equal(pipe.score(X, y), cached_pipe_2.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe_2.named_steps["transf_2"].means_,
        )
        assert cached_pipe_2.named_steps["transf_2"].timestamp_ == expected_ts
    finally:
        shutil.rmtree(cachedir)


def test_pipeline_none_classifier():
    # Test pipeline using None as preprocessing step and a classifier
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )
    clf = LogisticRegression(solver="lbfgs", random_state=0)
    pipe = make_pipeline(None, clf)
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.decision_function(X)
    pipe.score(X, y)


def test_pipeline_none_transformer():
    # Test pipeline using None and a transformer that implements transform and
    # inverse_transform
    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=0,
    )

    pca = PCA(whiten=True)
    pipe = make_pipeline(None, pca)
    pipe.fit(X, y)
    X_trans = pipe.transform(X)
    X_inversed = pipe.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_inversed)


def test_make_pipeline_memory():
    cachedir = mkdtemp()
    try:
        memory = Memory(cachedir, verbose=10)
        pipeline = make_pipeline(DummyTransf(), SVC(gamma="scale"), memory=memory)
        assert pipeline.memory is memory
        pipeline = make_pipeline(DummyTransf(), SVC(gamma="scale"))
        assert pipeline.memory is None
    finally:
        shutil.rmtree(cachedir)


def test_predict_with_predict_params():
    # tests that Pipeline passes predict_params to the final estimator
    # when predict is invoked
    pipe = Pipeline([("transf", Transf()), ("clf", DummyEstimatorParams())])
    pipe.fit(None, None)
    pipe.predict(X=None, got_attribute=True)
    assert pipe.named_steps["clf"].got_attribute


def test_score_samples_on_pipeline_without_score_samples():
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    # Test that a pipeline does not have score_samples method when the final
    # step of the pipeline does not have score_samples defined.
    pipe = make_pipeline(LogisticRegression())
    pipe.fit(X, y)

    inner_msg = "'LogisticRegression' object has no attribute 'score_samples'"
    outer_msg = "'Pipeline' has no attribute 'score_samples'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        pipe.score_samples(X)

    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)


def test_pipeline_param_error():
    clf = make_pipeline(LogisticRegression())
    with pytest.raises(
        ValueError,
        match="Pipeline.fit does not accept the sample_weight parameter",
    ):
        clf.fit([[0], [0]], [0, 1], sample_weight=[1, 1])


parameter_grid_test_verbose = (
    (est, pattern, method)
    for (est, pattern), method in itertools.product(
        [
            (
                Pipeline([("transf", Transf()), ("clf", FitParamT())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline([("transf", Transf()), ("noop", None), ("clf", FitParamT())]),
                r"\[Pipeline\].*\(step 1 of 3\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 3\) Processing noop.* total=.*\n"
                r"\[Pipeline\].*\(step 3 of 3\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline(
                    [
                        ("transf", Transf()),
                        ("noop", "passthrough"),
                        ("clf", FitParamT()),
                    ]
                ),
                r"\[Pipeline\].*\(step 1 of 3\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 3\) Processing noop.* total=.*\n"
                r"\[Pipeline\].*\(step 3 of 3\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline([("transf", Transf()), ("clf", None)]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline([("transf", None), ("mult", Mult())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing mult.* total=.*\n$",
            ),
            (
                Pipeline([("transf", "passthrough"), ("mult", Mult())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing mult.* total=.*\n$",
            ),
            (
                FeatureUnion([("mult1", Mult()), ("mult2", Mult())]),
                r"\[FeatureUnion\].*\(step 1 of 2\) Processing mult1.* total=.*\n"
                r"\[FeatureUnion\].*\(step 2 of 2\) Processing mult2.* total=.*\n$",
            ),
            (
                FeatureUnion([("mult1", "drop"), ("mult2", Mult()), ("mult3", "drop")]),
                r"\[FeatureUnion\].*\(step 1 of 1\) Processing mult2.* total=.*\n$",
            ),
        ],
        ["fit", "fit_transform", "fit_predict"],
    )
    if hasattr(est, method)
    and not (
        method == "fit_transform"
        and hasattr(est, "steps")
        and isinstance(est.steps[-1][1], FitParamT)
    )
)


@pytest.mark.parametrize("est, pattern, method", parameter_grid_test_verbose)
def test_verbose(est, method, pattern, capsys):
    func = getattr(est, method)

    X = [[1, 2, 3], [4, 5, 6]]
    y = [[7], [8]]

    est.set_params(verbose=False)
    func(X, y)
    assert not capsys.readouterr().out, "Got output for verbose=False"

    est.set_params(verbose=True)
    func(X, y)
    assert re.match(pattern, capsys.readouterr().out)
