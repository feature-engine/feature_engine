import pytest
from pandas.testing import assert_frame_equal

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)


def test_is_dataframe(df_vartypes):
    assert_frame_equal(_is_dataframe(df_vartypes), df_vartypes)
    with pytest.raises(TypeError):
        assert _is_dataframe([1, 2, 4])


def test_check_input_matches_training_df(df_vartypes):
    with pytest.raises(ValueError):
        assert _check_input_matches_training_df(df_vartypes, 4)


def test_contains_na(df_na):
    with pytest.raises(ValueError):
        assert _check_contains_na(df_na, ["Name", "City"])
    with pytest.raises(ValueError):
        assert _check_contains_na(
            df_na.reindex(index=[1, 2, 3, 4, 5, None]), variables=["index"]
        )
