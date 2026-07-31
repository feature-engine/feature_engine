import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.wrappers import SklearnWrapper
from tests.estimator_checks.estimator_checks import (
    check_raises_error_when_input_not_a_df,
)
from tests.estimator_checks.fit_functionality_checks import check_feature_names_in
from tests.estimator_checks.non_fitted_error_checks import check_raises_non_fitted_error
from tests.estimator_checks.variable_selection_checks import (
    check_all_types_variables_assignment,
    check_numerical_variables_assignment,
)


def test_sklearn_transformer_wrapper():
    check_estimator(
        estimator=SklearnWrapper(transformer=SimpleImputer()),
        expected_failed_checks=SklearnWrapper(
            transformer=SimpleImputer()
        )._more_tags()["_xfail_checks"],
    )


@pytest.mark.parametrize(
    "estimator", [SklearnWrapper(transformer=OrdinalEncoder())]
)
def test_check_estimator_from_feature_engine(estimator):
    check_raises_non_fitted_error(estimator)
    check_raises_error_when_input_not_a_df(estimator)
    check_feature_names_in(estimator)


def test_check_variables_assignment():
    check_numerical_variables_assignment(
        SklearnWrapper(transformer=StandardScaler())
    )
    check_all_types_variables_assignment(
        SklearnWrapper(transformer=OrdinalEncoder())
    )


def test_raises_error_when_no_transformer_passed():
    # this transformer needs an estimator as an input param.
    with pytest.raises(TypeError):
        SklearnWrapper()


def test_return_empty():
    X = pd.DataFrame({"var_cat": ["A", "B", "A"]})

    transformer = SklearnWrapper(
        transformer=StandardScaler(), variables=None, return_empty=False
    )
    with pytest.raises(
        TypeError, match="No numerical variables found in this dataframe"
    ):
        transformer.fit(X)

    transformer = SklearnWrapper(
        transformer=StandardScaler(), variables=None, return_empty=True
    )
    with pytest.warns(
        UserWarning,
        match="No numerical variables found in this dataframe. "
        "Returning an empty list.",
    ):
        transformer.fit(X)
    assert transformer.variables_ == []

    # if return_empty=True, transformer should return same df
    # after transformation
    dft = transformer.transform(X)
    pd.testing.assert_frame_equal(dft, X)
    assert transformer.get_feature_names_out() == list(X.columns)

    # when wrapping a transformer that selects all variable types (e.g.
    # OrdinalEncoder), find_all_variables always finds at least the 1 column
    # present in a non-empty dataframe, so return_empty can't be exercised
    # this way; there is no dataframe that reaches the "no variables" branch.
