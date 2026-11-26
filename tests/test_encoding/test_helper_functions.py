import re

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

from feature_engine.encoding._helper_functions import (
    TARGET_NAME,
    add_target_to_X,
    check_parameter_unseen,
)
from tests.backend_helpers import frame_to_dict, make_series


@pytest.mark.parametrize("accepted", ["one", False, [1, 2], ("one", "two"), 1])
def test_raises_error_when_accepted_values_not_permitted(accepted):
    msg = "accepted_values should be a list of strings. " f" Got {accepted} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_parameter_unseen("zero", accepted)


@pytest.mark.parametrize("unseen", ["zero", "One", "", 1, None, ["one"], ("one",)])
def test_raises_error_when_unseen_not_in_accepted_values(unseen):
    msg = f"Parameter `unseen` takes only values one, two. Got {unseen} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_parameter_unseen(unseen, ["one", "two"])


@pytest.mark.parametrize("to_target", [list, np.array, "series"])
def test_add_target_to_X_pairs_rows_by_position(make_df, to_target):
    X = make_df({"var_A": ["a", "b", "c"]})
    values = [1, 0, 1]
    if to_target == "series":
        y = make_series(make_df, values)
    else:
        y = to_target(values)

    Xy = add_target_to_X(nw.from_native(X), y).to_native()

    assert isinstance(Xy, make_df)
    assert frame_to_dict(Xy) == {"var_A": ["a", "b", "c"], TARGET_NAME: values}


def test_add_target_to_X_keeps_the_pandas_index():
    X = pd.DataFrame({"var_A": ["a", "b", "c"]}, index=[12, 10, 11])
    Xy = add_target_to_X(nw.from_native(X), np.array([1, 0, 1])).to_native()
    assert Xy.index.tolist() == [12, 10, 11]
    assert Xy[TARGET_NAME].tolist() == [1, 0, 1]
