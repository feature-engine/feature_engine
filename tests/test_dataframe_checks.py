from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)

import pytest
from pandas.testing import assert_frame_equal


def test_is_dataframe(df_vartypes):
    assert_frame_equal(_is_dataframe(df_vartypes), df_vartypes)
    with pytest.raises(TypeError):
        assert _is_dataframe([1, 2, 4])


def test_check_input_matches_training_df(df_vartypes):
    with pytest.raises(ValueError):
        assert _check_input_matches_training_df(df_vartypes, 4)
