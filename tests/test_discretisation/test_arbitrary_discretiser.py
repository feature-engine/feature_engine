import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

from feature_engine.discretisation import ArbitraryDiscretiser


def test_arbitrary_discretiser():
    boston_dataset = load_boston()
    data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    user_dict = {"LSTAT": [0, 10, 20, 30, np.Inf]}

    data_t1 = data.copy()
    data_t2 = data.copy()
    data_t1["LSTAT"] = pd.cut(data["LSTAT"], bins=[0, 10, 20, 30, np.Inf])
    data_t2["LSTAT"] = pd.cut(data["LSTAT"], bins=[0, 10, 20, 30, np.Inf], labels=False)

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_object=False, return_boundaries=False
    )
    X = transformer.fit_transform(data)

    # init params
    assert transformer.variables == ["LSTAT"]
    assert transformer.return_object is False
    assert transformer.return_boundaries is False
    # fit params
    assert transformer.binner_dict_ == user_dict
    # transform params
    pd.testing.assert_frame_equal(X, data_t2)

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_object=False, return_boundaries=True
    )
    X = transformer.fit_transform(data)
    pd.testing.assert_frame_equal(X, data_t1)
