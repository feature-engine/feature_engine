import pandas as pd
import pytest

from feature_engine.imputation import BaseImputer


@pytest.mark.parametrize("missing_only", [3, "lamp", (True, False), [1, True], 84.8])
def test_raises_error_when_missing_only_not_bool(missing_only):
    with pytest.raises(ValueError):
        BaseImputer(missing_only=missing_only)