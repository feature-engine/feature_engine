import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from feature_engine.discretisation.base_discretiser import BaseDiscretiser


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
        california_dataset = fetch_california_housing()
        data = pd.DataFrame(
            california_dataset.data, columns=california_dataset.feature_names
        )
        self.variables_ = ["HouseAge"]
        self.binner_dict_ = {"HouseAge": [0, 20, 40, 60, np.inf]}
        self.n_features_in_ = data.shape[1]
        self.feature_names_in_ = california_dataset.feature_names
        return self


def test_transform():
    california_dataset = fetch_california_housing()
    data = pd.DataFrame(
        california_dataset.data, columns=california_dataset.feature_names
    )

    data_t1 = data.copy()
    data_t2 = data.copy()

    # HouseAge is the median house age in the block group.
    data_t1["HouseAge"] = pd.cut(
        data["HouseAge"], bins=[0, 20, 40, 60, np.inf], include_lowest=True
    )
    data_t1["HouseAge"] = data_t1["HouseAge"].astype(str)
    data_t2["HouseAge"] = pd.cut(
        data["HouseAge"],
        bins=[0, 20, 40, 60, np.inf],
        labels=False,
        include_lowest=True,
    )

    transformer = MockClassFit(return_boundaries=False)
    X = transformer.fit_transform(data)
    pd.testing.assert_frame_equal(X, data_t2)

    transformer = MockClassFit(return_object=False, return_boundaries=True)
    X = transformer.fit_transform(data)
    pd.testing.assert_frame_equal(X, data_t1)
