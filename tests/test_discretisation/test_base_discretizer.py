import re

import numpy as np
import pandas as pd
import pytest

from feature_engine.discretisation.base_discretiser import BaseDiscretiser
from tests.backend_helpers import frame_to_dict

BINS = [0, 20, 40, 60, np.inf]


# init parameters
@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2, None])
def test_raises_error_when_return_object_not_bool(param):
    msg = f"return_object must be True or False. Got {param} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseDiscretiser(return_object=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2, None])
def test_raises_error_when_return_boundaries_not_bool(param):
    msg = f"return_boundaries must be True or False. Got {param} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseDiscretiser(return_boundaries=param)


@pytest.mark.parametrize(
    "param", [0.1, "hola", (True, False), {"a": True}, 0, -1, None]
)
def test_raises_error_when_precision_not_int(param):
    msg = f"precision must be a positive integer. Got {param} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseDiscretiser(precision=param)


@pytest.mark.parametrize(
    "return_object, return_boundaries, precision",
    [(False, False, 1), (True, False, 10), (False, True, 3)],
)
def test_init_param_assignment(return_object, return_boundaries, precision):
    transformer = BaseDiscretiser(
        return_object=return_object,
        return_boundaries=return_boundaries,
        precision=precision,
    )
    assert transformer.return_object is return_object
    assert transformer.return_boundaries is return_boundaries
    assert transformer.precision == precision


# fit and transform
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
