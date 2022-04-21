import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import TargetMeanDiscretiser


def test_equal_frequency_automatically_find_variables_and_return_as_numeric(
        df_normal_dist
):
    # fit discretiser and transform dataset
    transformer = TargetMeanDiscretiser(
        strategy="equal_frequency", bins=10, variables=None, return_object=False
    )
    X = transformer.fit_transform(df_normal_dist)

    # fit parameters
    _, bins = pd.cut(x=df_normal_dist["var"], bins=10, retbins=True, duplicates="drop")
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")

    # transform output
    X_t = [x for x in range(0, 10)]

    # test init params
    assert transformer.bins == 10
    assert transformer.variables is None
    assert transformer.return_object is None
    # test fit attr
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    # test transform output
    assert (transformer.binner_dict_["var"] == bins).all()
    assert all(x for x in X["var"].unique() if x not in X_t)
    # in equal-frequency discretisation, all intervals have the same proportion of values
    assert len((X["var"].value_counts()).unqiue()) == 1


def test_equal_width_automatically_find_variables_and_return_as_numeric(
        df_normal_dist
):
    transformer = TargetMeanDiscretiser(
        strategy="equal_width", bins=10, variables=None, return_object=False
    )
    X = transformer.fit_transform(df_normal_dist)

    # fit parameters
    _, bins = pd.qcut(x=df_normal_dist["var"], q=10, retbins=True, duplicates="drop")
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")

    # transform output
    X_t = [x for x in range(0, 10)]
    val_counts = [18, 17, 16, 13, 11, 7, 7, 5, 5, 1]

    # init params
    assert transformer.bins == 10
    assert transformer.variables is None
    assert transformer.return_object is False
    # fit params
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    # transform params
    assert (transformer.binner_dict_["var"] == bins).all()
    assert all(x for x in X["var"].unique() if x not in X_t)
    # in equal-width discretisation, intervals have number of values
    assert all(x for x in ["var"].value_counts() if x not in val_counts)


def test_automatically_find_variables_and_return_as_object(df_normal_dist):
    # equal-frequency
    transformer = TargetMeanDiscretiser(
        strategy="equal_frequency", bins=10, variables=None, return_object=True
    )
    X = transformer.fit_transform(df_normal_dist)
    assert X["var"].dtypes == "O"
