import pytest
from pandas.testing import assert_frame_equal

from feature_engine.utils import _define_variables, _find_numerical_variables, _find_categorical_variables
from feature_engine.utils import _is_dataframe, _check_input_matches_training_df


def test_define_variables():
    vars_ls = ['var1', 'var2', 'var1']
    vars_none = None
    vars_str = 'var1'
    assert _define_variables(vars_ls) == vars_ls
    assert _define_variables(vars_none) == vars_none
    assert _define_variables(vars_str) == [vars_str]


def test_find_numerical_variables(dataframe_vartypes):
    vars_num = ['Age', 'Marks']
    vars_mix = ['Age', 'Marks', 'Name']
    vars_none = None
    assert _find_numerical_variables(dataframe_vartypes, vars_num) == vars_num
    assert _find_numerical_variables(dataframe_vartypes, vars_none) == vars_num
    with pytest.raises(TypeError):
        assert _find_numerical_variables(dataframe_vartypes, vars_mix)


def test_find_categorical_variables(dataframe_vartypes):
    vars_cat = ['Name', 'City']
    vars_mix = ['Age', 'Marks', 'Name']
    vars_none = None
    assert _find_categorical_variables(dataframe_vartypes, vars_cat) == vars_cat
    assert _find_categorical_variables(dataframe_vartypes, vars_none) == vars_cat
    with pytest.raises(TypeError):
        assert _find_categorical_variables(dataframe_vartypes, vars_mix)


def test_is_dataframe(dataframe_vartypes):
    assert_frame_equal(_is_dataframe(dataframe_vartypes), dataframe_vartypes)
    with pytest.raises(TypeError):
        assert _is_dataframe([1, 2, 4])


def test_check_input_matches_training_df(dataframe_vartypes):
    with pytest.raises(ValueError):
        assert _check_input_matches_training_df(dataframe_vartypes, 4)
