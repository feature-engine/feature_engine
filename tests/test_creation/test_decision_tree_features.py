import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from feature_engine.creation import DecisionTreeFeatures


@pytest.fixture(scope="module")
def df_creation():
    data = {
        "Name": [
            "tom",
            "nick",
            "krish",
            "megan",
            "peter",
            "jordan",
            "fred",
            "sam",
            "alexa",
            "brittany",
        ],
        "Age": [20, 44, 19, 33, 51, 40, 41, 37, 30, 54],
        "Height": [164, 150, 178, 158, 188, 190, 168, 174, 176, 171],
        "Marks": [1.0, 0.8, 0.6, 0.1, 0.3, 0.4, 0.8, 0.6, 0.5, 0.2],
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def regression_target():
    return pd.Series([4.1, 5.8, 3.9, 6.2, 4.3, 4.5, 7.2, 4.4, 4.1, 6.7])


@pytest.fixture(scope="module")
def classification_target():
    return pd.Series([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])


@pytest.fixture(scope="module")
def multiclass_target():
    return pd.Series([1, 1, 2, 2, 0, 1, 0, 1, 0, 0])


@pytest.mark.parametrize("precision", ["string", 0.1, -1, np.nan])
def test_error_if_precision_gets_not_permitted_value(precision):
    msg = "precision must be None or a positive integer. " f"Got {precision} instead."
    with pytest.raises(ValueError, match=msg):
        DecisionTreeFeatures(precision=precision)


@pytest.mark.parametrize("regression", ["string", 0.1, -1, np.nan])
def test_error_if_regression_gets_not_permitted_value(regression):
    msg = f"regression must be a boolean value. Got {regression} instead."
    with pytest.raises(ValueError, match=msg):
        DecisionTreeFeatures(regression=regression)


@pytest.mark.parametrize("drop", ["string", 0.1, -1, np.nan])
def test_error_if_drop_original_gets_not_permitted_value(drop):
    msg = (
        "drop_original takes only boolean values True and False. "
        f"Got {drop} instead."
    )
    with pytest.raises(ValueError, match=msg):
        DecisionTreeFeatures(drop_original=drop)


@pytest.mark.parametrize(
    "input_features, expected",
    [
        (1, ["vara", "varb", "varc"]),
        (
            2,
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
            ],
        ),
        (
            3,
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
        (
            4,
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
        (
            100,
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
    ],
)
def test_create_variable_combinations_when_int(input_features, expected):
    vars = ["vara", "varb", "varc"]
    transformer = DecisionTreeFeatures()
    combos = transformer._create_variable_combinations(
        variables=vars, how_to_combine=input_features
    )
    assert combos == expected


@pytest.mark.parametrize(
    "vars, expected",
    [
        (
            ["vara", "varb", "varc"],
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
        (["vara", "varb"], ["vara", "varb", ["vara", "varb"]]),
        (["vara"], ["vara"]),
    ],
)
def test_create_variable_combinations_when_None(vars, expected):
    transformer = DecisionTreeFeatures()
    combos = transformer._create_variable_combinations(
        variables=vars, how_to_combine=None
    )
    assert combos == expected


@pytest.mark.parametrize(
    "input_features, expected",
    [
        (
            [2, 3],
            [
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
        ([1, 3], ["vara", "varb", "varc", ["vara", "varb", "varc"]]),
    ],
)
def test_create_variable_combinations_when_list(input_features, expected):
    vars = ["vara", "varb", "varc"]
    transformer = DecisionTreeFeatures()
    combos = transformer._create_variable_combinations(
        variables=vars, how_to_combine=input_features
    )
    assert combos == expected


@pytest.mark.parametrize(
    "input_features, expected",
    [
        (
            (("vara", "varb"), ("vara"), ("vara", "varb", "varc")),
            [["vara", "varb"], "vara", ["vara", "varb", "varc"]],
        ),
        ((("vara", "varc"), ("vara", "varb")), [["vara", "varc"], ["vara", "varb"]]),
    ],
)
def test_create_variable_combinations_when_tuple(input_features, expected):
    vars = ["vara", "varb", "varc"]
    transformer = DecisionTreeFeatures()
    combos = transformer._create_variable_combinations(
        variables=vars, how_to_combine=input_features
    )
    assert combos == expected


def test_feature_creation_regression(df_creation, regression_target):
    X = df_creation.copy()
    y = regression_target.copy()

    scoring = "neg_mean_squared_error"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs)
    Xt = tr.fit_transform(X, y)

    # get expected
    est = DecisionTreeRegressor(random_state=rs)
    tree = GridSearchCV(
        est,
        cv=3,
        scoring=scoring,
        param_grid={"max_depth": [1, 2, 3, 4]},
    )

    combos = [
        "Age",
        "Height",
        "Marks",
        ["Age", "Height"],
        ["Age", "Marks"],
        ["Height", "Marks"],
        ["Age", "Height", "Marks"],
    ]
    var_names = [f"tree({item})" for item in combos]

    X_exp = df_creation.copy()
    for i in range(len(combos)):
        varn = var_names[i]
        combon = combos[i]
        if isinstance(combon, str):
            tree.fit(X[combon].to_frame(), y)
            X_exp[varn] = tree.predict(X[combon].to_frame())
        else:
            tree.fit(X[combon], y)
            X_exp[varn] = tree.predict(X[combon])

    pd.testing.assert_frame_equal(Xt, X_exp)


def test_feature_creation_regression_and_precision(df_creation, regression_target):
    X = df_creation.copy()
    y = regression_target.copy()

    scoring = "neg_mean_squared_error"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs, precision=1)
    Xt = tr.fit_transform(X, y)

    # get expected
    est = DecisionTreeRegressor(random_state=rs)
    tree = GridSearchCV(
        est,
        cv=3,
        scoring=scoring,
        param_grid={"max_depth": [1, 2, 3, 4]},
    )

    combos = [
        "Age",
        "Height",
        "Marks",
        ["Age", "Height"],
        ["Age", "Marks"],
        ["Height", "Marks"],
        ["Age", "Height", "Marks"],
    ]
    var_names = [f"tree({item})" for item in combos]

    X_exp = df_creation.copy()
    for i in range(len(combos)):
        varn = var_names[i]
        combon = combos[i]
        if isinstance(combon, str):
            tree.fit(X[combon].to_frame(), y)
            preds = tree.predict(X[combon].to_frame())
            X_exp[varn] = np.round(preds, 1)
        else:
            tree.fit(X[combon], y)
            preds = tree.predict(X[combon])
            X_exp[varn] = np.round(preds, 1)

    pd.testing.assert_frame_equal(Xt, X_exp)


def test_feature_creation_regression_drop_original(df_creation, regression_target):
    X = df_creation.copy()
    y = regression_target.copy()

    scoring = "neg_mean_squared_error"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs, drop_original=True)
    Xt = tr.fit_transform(X, y)

    # get expected
    est = DecisionTreeRegressor(random_state=rs)
    tree = GridSearchCV(
        est,
        cv=3,
        scoring=scoring,
        param_grid={"max_depth": [1, 2, 3, 4]},
    )

    combos = [
        "Age",
        "Height",
        "Marks",
        ["Age", "Height"],
        ["Age", "Marks"],
        ["Height", "Marks"],
        ["Age", "Height", "Marks"],
    ]
    var_names = [f"tree({item})" for item in combos]

    X_exp = df_creation.copy()
    for i in range(len(combos)):
        varn = var_names[i]
        combon = combos[i]
        if isinstance(combon, str):
            tree.fit(X[combon].to_frame(), y)
            X_exp[varn] = tree.predict(X[combon].to_frame())
        else:
            tree.fit(X[combon], y)
            X_exp[varn] = tree.predict(X[combon])
    X_exp.drop(["Age", "Height", "Marks"], axis=1, inplace=True)

    pd.testing.assert_frame_equal(Xt, X_exp)


def test_feature_creation_binary_classif(df_creation, classification_target):
    X = df_creation.copy()
    y = classification_target.copy()

    scoring = "roc_auc"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs, regression=False)
    Xt = tr.fit_transform(X, y)

    # get expected
    est = DecisionTreeClassifier(random_state=rs)
    tree = GridSearchCV(
        est,
        cv=3,
        scoring=scoring,
        param_grid={"max_depth": [1, 2, 3, 4]},
    )

    combos = [
        "Age",
        "Height",
        "Marks",
        ["Age", "Height"],
        ["Age", "Marks"],
        ["Height", "Marks"],
        ["Age", "Height", "Marks"],
    ]
    var_names = [f"tree({item})" for item in combos]

    X_exp = df_creation.copy()
    for i in range(len(combos)):
        varn = var_names[i]
        combon = combos[i]
        if isinstance(combon, str):
            tree.fit(X[combon].to_frame(), y)
            preds = tree.predict_proba(X[combon].to_frame())
            X_exp[varn] = preds[:, 1]
        else:
            tree.fit(X[combon], y)
            preds = tree.predict_proba(X[combon])
            X_exp[varn] = preds[:, 1]

    pd.testing.assert_frame_equal(Xt, X_exp)


def test_feature_creation_binary_classif_w_precision(
    df_creation, classification_target
):
    X = df_creation.copy()
    y = classification_target.copy()

    scoring = "roc_auc"
    rs = 0
    tr = DecisionTreeFeatures(
        scoring=scoring, random_state=rs, regression=False, precision=2
    )
    Xt = tr.fit_transform(X, y)

    # get expected
    est = DecisionTreeClassifier(random_state=rs)
    tree = GridSearchCV(
        est,
        cv=3,
        scoring=scoring,
        param_grid={"max_depth": [1, 2, 3, 4]},
    )

    combos = [
        "Age",
        "Height",
        "Marks",
        ["Age", "Height"],
        ["Age", "Marks"],
        ["Height", "Marks"],
        ["Age", "Height", "Marks"],
    ]
    var_names = [f"tree({item})" for item in combos]

    X_exp = df_creation.copy()
    for i in range(len(combos)):
        varn = var_names[i]
        combon = combos[i]
        if isinstance(combon, str):
            tree.fit(X[combon].to_frame(), y)
            preds = tree.predict_proba(X[combon].to_frame())
            X_exp[varn] = np.round(preds[:, 1], 2)
        else:
            tree.fit(X[combon], y)
            preds = tree.predict_proba(X[combon])
            X_exp[varn] = np.round(preds[:, 1], 2)

    pd.testing.assert_frame_equal(Xt, X_exp)


def test_feature_creation_binary_multiclass(df_creation, multiclass_target):
    X = df_creation.copy()
    y = multiclass_target.copy()

    scoring = "roc_auc"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs, regression=False)
    Xt = tr.fit_transform(X, y)

    # get expected
    est = DecisionTreeClassifier(random_state=rs)
    tree = GridSearchCV(
        est,
        cv=3,
        scoring=scoring,
        param_grid={"max_depth": [1, 2, 3, 4]},
    )

    combos = [
        "Age",
        "Height",
        "Marks",
        ["Age", "Height"],
        ["Age", "Marks"],
        ["Height", "Marks"],
        ["Age", "Height", "Marks"],
    ]
    var_names = [f"tree({item})" for item in combos]

    X_exp = df_creation.copy()
    for i in range(len(combos)):
        varn = var_names[i]
        combon = combos[i]
        if isinstance(combon, str):
            tree.fit(X[combon].to_frame(), y)
            preds = tree.predict(X[combon].to_frame())
            X_exp[varn] = preds
        else:
            tree.fit(X[combon], y)
            preds = tree.predict(X[combon])
            X_exp[varn] = preds

    pd.testing.assert_frame_equal(Xt, X_exp)


def test_get_feature_names_out(df_creation, regression_target):
    X = df_creation.copy()
    y = regression_target.copy()

    tr = DecisionTreeFeatures(
        variables=["Age", "Marks"],
    )

    Xt = tr.fit_transform(X, y)
    feat_out = Xt.columns.to_list()
    assert tr.get_feature_names_out() == feat_out
    assert tr.get_feature_names_out(X.columns.to_list()) == feat_out


def test_get_feature_names_out_from_pipeline(df_creation, regression_target):
    X = df_creation.copy()
    y = regression_target.copy()

    # set up transformer
    tr = DecisionTreeFeatures(
        variables=["Age", "Marks"],
    )

    pipe = Pipeline([("transformer", tr)])

    Xt = pipe.fit_transform(X, y)
    feat_out = Xt.columns.to_list()
    assert pipe.get_feature_names_out(input_features=None) == feat_out
    assert pipe.get_feature_names_out(input_features=X.columns.to_list()) == feat_out


@pytest.mark.parametrize("_input_features", ["hola", ["Age", "Marks"]])
def test_get_feature_names_out_raises_error_when_wrong_param(
    _input_features, df_creation, regression_target
):
    X = df_creation.copy()
    y = regression_target.copy()

    tr = DecisionTreeFeatures(
        variables=["Age", "Marks"],
    )
    tr.fit(X, y)

    with pytest.raises(ValueError):
        tr.get_feature_names_out(input_features=_input_features)


def test_error_when_regression_true_and_target_binary(
    df_creation, classification_target
):
    X = df_creation.copy()
    y = classification_target.copy()
    tr = DecisionTreeFeatures(regression=True)

    msg = (
        "Trying to fit a regression to a binary target is not "
        + "allowed by this transformer. Check the target values "
        + "or set regression to False."
    )
    with pytest.raises(ValueError, match=msg):
        tr.fit(X, y)


def test_user_enter_param_grid(df_creation, classification_target):
    X = df_creation.copy()
    y = classification_target.copy()
    scoring = "roc_auc"
    rs = 0
    grid = {"max_depth": [1, 2, 3, 4]}
    tr = DecisionTreeFeatures(
        scoring=scoring, random_state=rs, regression=False, param_grid=grid
    )
    Xt = tr.fit_transform(X, y)

    # get expected
    est = DecisionTreeClassifier(random_state=rs)
    tree = GridSearchCV(
        est,
        cv=3,
        scoring=scoring,
        param_grid={"max_depth": [1, 2, 3, 4]},
    )

    combos = [
        "Age",
        "Height",
        "Marks",
        ["Age", "Height"],
        ["Age", "Marks"],
        ["Height", "Marks"],
        ["Age", "Height", "Marks"],
    ]
    var_names = [f"tree({item})" for item in combos]

    X_exp = df_creation.copy()
    for i in range(len(combos)):
        varn = var_names[i]
        combon = combos[i]
        if isinstance(combon, str):
            tree.fit(X[combon].to_frame(), y)
            preds = tree.predict_proba(X[combon].to_frame())
            X_exp[varn] = preds[:, 1]
        else:
            tree.fit(X[combon], y)
            preds = tree.predict_proba(X[combon])
            X_exp[varn] = preds[:, 1]

    pd.testing.assert_frame_equal(Xt, X_exp)
