import numpy as np
import pandas as pd

from feature_engine._base_transformers.mixins import TransformXyMixin


class MockTransformer(TransformXyMixin):
    def transform(self, X):
        return X.iloc[1:-1].copy()


def test_transform_x_y_method(df_vartypes):
    # single target
    y = pd.Series(0, index=np.arange(len(df_vartypes)))
    transformer = MockTransformer()
    Xt, yt = transformer.transform_x_y(df_vartypes, y)

    assert len(Xt) == len(yt)
    assert len(Xt) != len(df_vartypes)
    assert len(yt) != len(y)
    assert (Xt.index == yt.index).all()
    assert (Xt.index == [1, 2]).all()

    # multioutput target
    y = (
        pd.DataFrame(columns=["vara", "varb"], index=df_vartypes.index)
        .astype(float)
        .fillna(0)
    )
    Xt, yt = transformer.transform_x_y(df_vartypes, y)

    assert len(Xt) == len(yt)
    assert len(Xt) != len(df_vartypes)
    assert len(yt) != len(y)
    assert (Xt.index == yt.index).all()
    assert (Xt.index == [1, 2]).all()
