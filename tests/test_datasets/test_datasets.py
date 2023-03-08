from feature_engine.datasets import load_titanic
import pytest


def test_load_titanic():
    data = load_titanic()

    assert data.shape == (1309, 14)


@pytest.mark.parametrize(
    "handle_missing",
    [True, False],
)
def test_titanic_datatypes(handle_missing):
    data = load_titanic(handle_missing=handle_missing)
    num_columns = ["pclass", "survived", "age", "sibsp", "parch", "fare"]
    assert data[num_columns].dtypes.astype(str).isin(["int64", "float64"]).all()


@pytest.mark.parametrize(
    "predictors_only,expected_X,expected_y",
    [(True, (1309, 7), (1309,)), (False, (1309, 13), (1309,))],
)
def test_load_titanic_return_X_y(predictors_only, expected_X, expected_y):
    X, y = load_titanic(return_X_y_frame=True, predictors_only=predictors_only)

    assert X.shape == expected_X
    assert y.shape == expected_y


@pytest.mark.parametrize(
    "handle_missing,predictors_only, null_sum",
    [(True, True, 0), (True, False, 0), (False, True, 266), (False, False, 3855)],
)
def test_load_titanic_raw(handle_missing, predictors_only, null_sum):
    X, y = load_titanic(
        return_X_y_frame=True,
        predictors_only=predictors_only,
        handle_missing=handle_missing,
    )

    assert X.isnull().sum().sum() == null_sum
