import pandas as pd
import pytest
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from feature_engine.wrappers import SklearnTransformerWrapper
from tests.estimator_checks.estimator_checks import (
    check_raises_error_when_input_not_a_df,
)
from tests.estimator_checks.fit_functionality_checks import check_feature_names_in
from tests.estimator_checks.non_fitted_error_checks import check_raises_non_fitted_error
from tests.estimator_checks.variable_selection_checks import (
    check_all_types_variables_assignment,
    check_numerical_variables_assignment,
)

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

if sklearn_version < parse_version("1.6"):

    def test_sklearn_transformer_wrapper():
        check_estimator(SklearnTransformerWrapper(transformer=SimpleImputer()))

else:

    def test_sklearn_transformer_wrapper():
        check_estimator(
            estimator=SklearnTransformerWrapper(transformer=SimpleImputer()),
            expected_failed_checks=SklearnTransformerWrapper(
                transformer=SimpleImputer()
            )._more_tags()["_xfail_checks"],
        )


@pytest.mark.parametrize(
    "estimator", [SklearnTransformerWrapper(transformer=OrdinalEncoder())]
)
def test_check_estimator_from_feature_engine(estimator):
    check_raises_non_fitted_error(estimator)
    check_raises_error_when_input_not_a_df(estimator)
    check_feature_names_in(estimator)


def test_check_variables_assignment():
    check_numerical_variables_assignment(
        SklearnTransformerWrapper(transformer=StandardScaler())
    )
    check_all_types_variables_assignment(
        SklearnTransformerWrapper(transformer=OrdinalEncoder())
    )


def test_raises_error_when_no_transformer_passed():
    # this transformer needs an estimator as an input param.
    with pytest.raises(TypeError):
        SklearnTransformerWrapper()


def test_return_empty():
    # SklearnTransformerWrapper is not part of the check_feature_engine_estimator
    # pipeline, so return_empty is tested directly here. The shared
    # check_return_empty helper isn't reused because, unlike feature-engine's own
    # transformers, this wrapper delegates the actual fitting to the wrapped
    # sklearn transformer. When the wrapped transformer is e.g. StandardScaler,
    # fitting it on zero columns raises its own (expected) error -- return_empty
    # only controls whether *variable selection* raises, not what the wrapped
    # transformer does afterwards with an empty selection.
    X = pd.DataFrame({"var_cat": ["A", "B", "A"]})

    transformer = SklearnTransformerWrapper(
        transformer=StandardScaler(), variables=None, return_empty=False
    )
    with pytest.raises(TypeError):
        transformer.fit(X)

    transformer = SklearnTransformerWrapper(
        transformer=StandardScaler(), variables=None, return_empty=True
    )
    with pytest.warns(UserWarning):
        transformer.fit(X)
    assert transformer.variables_ == []

    # if return_empty=True, transformer should return same df
    # after transformation
    dft = transformer.transform(X)
    pd.testing.assert_frame_equal(dft, X)

    # when wrapping a transformer that selects all variable types (e.g.
    # OrdinalEncoder), find_all_variables always finds at least the 1 column
    # present in a non-empty dataframe, so return_empty can't be exercised
    # this way; there is no dataframe that reaches the "no variables" branch.
