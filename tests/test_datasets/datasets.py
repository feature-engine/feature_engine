import pytest
from pandas import DataFrame

from feature_engine.datasets import load_titanic


def test_load_titanic():
    data = load_titanic()
    variables = [
        "pclass",
        "survived",
        "name",
        "sex",
        "age",
        "sibsp",
        "parch",
        "ticket",
        "fare",
        "cabin",
        "embarked",
        "boat",
        "body",
        "home.dest",
    ]
    assert isinstance(data, DataFrame)
    assert data.shape == (1309, 14)
    assert list(data.columns) == variables


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
    [(True, (1309, 8), (1309,)), (False, (1309, 13), (1309,))],
)
def test_return_X_y(predictors_only, expected_X, expected_y):
    X, y = load_titanic(return_X_y_frame=True, predictors_only=predictors_only)
    assert X.shape == expected_X
    assert y.shape == expected_y


@pytest.mark.parametrize(
    "handle_missing,predictors_only, null_sum",
    [(True, True, 0), (True, False, 0), (False, True, 1280), (False, False, 3855)],
)
def test_load_titanic_raw(handle_missing, predictors_only, null_sum):
    X, y = load_titanic(
        return_X_y_frame=True,
        predictors_only=predictors_only,
        handle_missing=handle_missing,
    )

    assert X.isnull().sum().sum() == null_sum


@pytest.mark.parametrize("cabin", [None, "letter_only", "drop"])
def test_cabin(cabin):

    data = load_titanic(cabin=None)
    assert "cabin" in data.columns
    assert list(data["cabin"].head(4).values) == ["B5", "C22 C26", "C22 C26", "C22 C26"]

    data = load_titanic(cabin="letter_only")
    assert list(data["cabin"].head(4).values) == ["B", "C", "C", "C"]

    data = load_titanic(cabin="drop")
    assert "cabin" not in data.columns


@pytest.mark.parametrize("param", ["Hello", [True], 1, 2.5])
def test_raise_value_error_if_not_boolean(param):
    with pytest.raises(ValueError):
        load_titanic(return_X_y_frame=param)
    with pytest.raises(ValueError):
        load_titanic(predictors_only=param)
    with pytest.raises(ValueError):
        load_titanic(handle_missing=param)


@pytest.mark.parametrize("cabin", ["Hello", True, 1, 2.5])
def test_cabin_raise_value_error(cabin):
    with pytest.raises(ValueError):
        load_titanic(cabin=cabin)
