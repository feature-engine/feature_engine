from feature_engine.datasets import load_titanic
import pytest


def test_load_titanic():
    data = load_titanic()

    assert data.shape == (1309, 14)


@pytest.mark.parametrize(
    "predictors_only,expected_X,expected_y",
    [(True, (1309, 7), (1309,)), (False, (1309, 13), (1309,))],
)
def test_load_titanic_return_X_y(predictors_only, expected_X, expected_y):
    X, y = load_titanic(return_X_y_frame=True, predictors_only=predictors_only)

    assert X.shape == expected_X
    assert y.shape == expected_y


@pytest.mark.parametrize(
    "raw,predictors_only, null_sum",
    [(True, True, 0), (True, False, 0), (False, True, 266), (False, False, 3855)],
)
def test_load_titanic_raw(raw, predictors_only, null_sum):
    X, y = load_titanic(return_X_y_frame=True, predictors_only=predictors_only, raw=raw)

    assert X.isnull().sum().sum() == null_sum
