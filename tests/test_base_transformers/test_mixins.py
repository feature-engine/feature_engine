import numpy as np
import pytest

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin


class MockClass(GetFeatureNamesOutMixin):
    def __init__(self):
        self.feature_names_in_ = ["var1", "var2", "var3"]


class MockClassInt(GetFeatureNamesOutMixin):
    def __init__(self):
        self.feature_names_in_ = [1, 2, 3]


@pytest.mark.parametrize(
    "input_features",
    [None, ["var1", "var2", "var3"], np.array(["var1", "var2", "var3"])],
)
def test_GetFeatureNamesOut_permitted_params(input_features):
    transformer = MockClass()
    assert (
        transformer.get_feature_names_out(input_features=input_features)
        == transformer.feature_names_in_
    )


@pytest.mark.parametrize("input_features", [None, [1, 2, 3], np.array([1, 2, 3])])
def test_GetFeatureNamesOut_permitted_params_int(input_features):
    transformer = MockClassInt()
    assert (
        transformer.get_feature_names_out(input_features=input_features)
        == transformer.feature_names_in_
    )


def test_GetFeatureNamesOut_non_permitted_params():
    transformer = MockClass()

    with pytest.raises(ValueError) as record:
        transformer.get_feature_names_out(input_features=["var1"])
    assert "feature_names_in_" in str(record)

    with pytest.raises(ValueError) as record:
        transformer.get_feature_names_out(input_features=np.array(["var1", "var2"]))
    assert "feature_names_in_" in str(record)

    with pytest.raises(ValueError) as record:
        transformer.get_feature_names_out(input_features="var1")
    assert "list or an array" in str(record)

    with pytest.raises(ValueError) as record:
        transformer.get_feature_names_out(input_features=True)
    assert "list or an array" in str(record)
