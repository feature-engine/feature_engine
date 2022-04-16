import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import YeoJohnsonTransformer


def test_automatically_select_variables(df_vartypes):
    # test case 1: automatically select variables
    transformer = YeoJohnsonTransformer(variables=None)
    X = transformer.fit_transform(df_vartypes)

    # expected result
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [10.167, 10.5406, 9.78774, 9.40229]
    transf_df["Marks"] = [0.804449, 0.722367, 0.638807, 0.553652]

    # test init params
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


def test_transformer_on_integer_variables():
    df = pd.DataFrame(
        {
            "var1": [0, 1, 0, 2, 3, 4, 5, 6, 8, 10],
            "var2": [12, 11, 10, 15, 13, 12, 11, 10, 10, 20],
        }
    )

    dft = pd.DataFrame(
        {
            "var1": {
                0: 0.0,
                1: 0.7871467037957388,
                2: 0.0,
                3: 1.34716625120788,
                4: 1.797027857352365,
                5: 2.1794549065159363,
                6: 2.5155129679774246,
                7: 2.817344570368886,
                8: 3.346739213848269,
                9: 3.8051709334268566,
            },
            "var2": {
                0: 0.2891005444159968,
                1: 0.2890875957028113,
                2: 0.2890687942494933,
                3: 0.2891213447054929,
                4: 0.2891097235906253,
                5: 0.2891005444159968,
                6: 0.2890875957028113,
                7: 0.2890687942494933,
                8: 0.2890687942494933,
                9: 0.28913341330818815,
            },
        }
    )

    X_tr = YeoJohnsonTransformer().fit_transform(df)
    pd.testing.assert_frame_equal(X_tr, dft)


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = YeoJohnsonTransformer()
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = YeoJohnsonTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = YeoJohnsonTransformer()
        transformer.transform(df_vartypes)
