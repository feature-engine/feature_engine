import pandas as pd
import pytest

from feature_engine.selection import SelectByInformationValue


@pytest.mark.parametrize(
    "_threshold", ["python", (True, False), [4.3, 3]]
)
def test_error_when_not_permitted_threshold(_threshold):
    with pytest.raises(ValueError):
        SelectByInformationValue(
            variables=None,
            threshold=_threshold,
            ignore_format=False,
            confirm_variables=False,
        )


def test_error_when_more_than_two_classes(df_enc_numeric):
    transformer = SelectByInformationValue(
        variables=None,
        threshold=0.2,
        ignore_format=False,
        confirm_variables=False
    )
    with pytest.raises(ValueError):
        transformer.fit(
            df_enc_numeric[["var_A", "target"]], df_enc_numeric["var_B"]
        )


