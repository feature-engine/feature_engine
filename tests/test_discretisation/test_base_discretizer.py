import numpy as np
import pandas as pd
import pytest

from feature_engine.discretisation.base_discretiser import BaseDiscretiser
from tests.backend_helpers import frame_to_dict

BINS = [0, 20, 40, 60, np.inf]


# test init params
@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_object_not_bool(param):
    with pytest.raises(ValueError):
        BaseDiscretiser(return_object=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_boundaries_not_bool(param):
    with pytest.raises(ValueError):
        BaseDiscretiser(return_boundaries=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 0, -1])
def test_raises_error_when_precision_not_int(param):
    with pytest.raises(ValueError):
        BaseDiscretiser(precision=param)


@pytest.mark.parametrize("params", [(False, 1), (True, 10)])
def test_correct_param_assignment_at_init(params):
    param1, param2 = params
    t = BaseDiscretiser(
        return_object=param1, return_boundaries=param1, precision=param2
    )
    assert t.return_object is param1
    assert t.return_boundaries is param1
    assert t.precision == param2


class MockClassFit(BaseDiscretiser):
    def fit(self, X):
        # bins are hard-coded rather than learnt, so this mock works unchanged
        # on both pandas and polars input.
        self.variables_ = ["HouseAge"]
        self.binner_dict_ = {"HouseAge": BINS}
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        return self


def test_transform(make_df, data_california):
    # ground truth via pandas.cut: bins are fixed by MockClassFit, so both
    # backends must reproduce this exact output.
    house_age = pd.Series(data_california["HouseAge"])
    expected_codes = pd.cut(
        house_age, bins=BINS, labels=False, include_lowest=True
    ).tolist()
    expected_labels = (
        pd.cut(house_age, bins=BINS, include_lowest=True).astype(str).tolist()
    )

    data = make_df(data_california)

    transformer = MockClassFit(return_boundaries=False)
    X = transformer.fit_transform(data)
    assert isinstance(X, make_df)
    assert frame_to_dict(X)["HouseAge"] == expected_codes

    transformer = MockClassFit(return_object=False, return_boundaries=True)
    X = transformer.fit_transform(data)
    assert isinstance(X, make_df)
    assert frame_to_dict(X)["HouseAge"] == expected_labels
