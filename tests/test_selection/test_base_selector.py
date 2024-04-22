import pytest
from pandas.testing import assert_frame_equal

from feature_engine.selection.base_selector import BaseSelector


@pytest.mark.parametrize("val", [None, "hola", [True]])
def test_confirm_variables_in_init(val):
    with pytest.raises(ValueError):
        BaseSelector(confirm_variables=val)


class MockClass(BaseSelector):
    def __init__(self, variables=None, confirm_variables=False):
        self.variables = variables
        self.confirm_variables = confirm_variables

    def fit(self, X, y=None):
        self.features_to_drop_ = ["Name", "Marks"]
        self._get_feature_names_in(X)
        return self


def test_transform_method(df_vartypes):
    transformer = MockClass()
    transformer.fit(df_vartypes)
    Xt = transformer.transform(df_vartypes)

    # tests output of transform
    assert_frame_equal(Xt, df_vartypes.drop(["Name", "Marks"], axis=1))

    # tests this line: X = X[self.feature_names_in_]
    assert_frame_equal(
        transformer.transform(df_vartypes[["City", "Age", "Name", "Marks", "dob"]]),
        Xt,
    )
    # test error when there is a df shape missmatch
    with pytest.raises(ValueError):
        assert transformer.transform(df_vartypes[["Age", "Marks"]])


def test_get_feature_names_in(df_vartypes):
    tr = MockClass()
    tr._get_feature_names_in(df_vartypes)
    assert tr.n_features_in_ == df_vartypes.shape[1]
    assert tr.feature_names_in_ == list(df_vartypes.columns)


def test_get_support(df_vartypes):
    tr = MockClass()
    tr.fit(df_vartypes)
    v_bool = [False, True, True, False, True]
    v_ind = [1, 2, 4]
    assert tr.get_support() == v_bool
    assert list(tr.get_support(indices=True)) == v_ind
