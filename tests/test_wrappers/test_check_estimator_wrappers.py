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
