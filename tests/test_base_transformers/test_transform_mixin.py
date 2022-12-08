import pandas as pd

from feature_engine._base_transformers.mixins import TransformerMixin


class MockClass(TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        X["var_sum"] = X.sum(axis=1)
        return X


def test_transformer_mixin():
    df = pd.DataFrame(
        {
            "var_1": [0, 1, 2, 3],
            "var_2": [0, 1, 2, 3],
        }
    )

    df_exp = pd.DataFrame(
        {
            "var_1": [0, 1, 2, 3],
            "var_2": [0, 1, 2, 3],
            "var_sum": [0, 2, 4, 6],
        }
    )

    transformer = MockClass()
    dft = transformer.fit_transform(df)

    assert transformer.feature_names_in_ == ["var_1", "var_2"]
    assert transformer.n_features_in_ == 2
    pd.testing.assert_frame_equal(dft, df_exp)

    y = pd.Series([0, 1, 0, 0])
    transformer = MockClass()
    dft = transformer.fit_transform(df, y)

    assert transformer.feature_names_in_ == ["var_1", "var_2"]
    assert transformer.n_features_in_ == 2
    pd.testing.assert_frame_equal(dft, df_exp)
