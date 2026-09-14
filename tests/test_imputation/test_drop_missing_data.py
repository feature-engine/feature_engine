import pytest

from feature_engine.imputation import DropMissingData
from tests.backend_helpers import make_series, null_count, to_dict


def test_detect_variables_with_na(make_df, data_na_dob):
    # test case 1: automatically detect variables with missing data
    imputer = DropMissingData(missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(make_df(data_na_dob))
    # init params
    assert imputer.missing_only is True
    assert imputer.threshold is None
    assert imputer.variables is None
    # fit params
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 6
    # transform outputs: only rows complete in variables_ survive
    assert isinstance(X_transformed, make_df)
    assert X_transformed.shape == (5, 6)
    assert to_dict(X_transformed)["Age"] == [20, 21, 23, 41, 37]
    for var in imputer.variables_:
        assert null_count(X_transformed, var) == 0


def test_transform_x_y(make_df, data_na_dob):
    df_na = make_df(data_na_dob)
    y = make_series(make_df, list(range(8)))
    imputer = DropMissingData(missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(df_na)
    assert X_transformed.shape == (5, 6)
    assert len(X_transformed) != len(y)

    Xt, yt = imputer.transform_x_y(df_na, y)
    assert isinstance(Xt, make_df)
    assert isinstance(yt, type(y))
    # rows 0, 1, 4, 6, 7 are the ones complete in Name/City/Studies/Age/Marks
    assert list(yt) == [0, 1, 4, 6, 7]
    assert to_dict(Xt)["Age"] == [20, 21, 23, 41, 37]
    assert len(Xt) == len(yt)
    assert len(df_na) != len(Xt)


def test_selelct_all_variables_when_variables_is_none(make_df, data_na_dob):
    imputer = DropMissingData(missing_only=False, variables=None)
    X_transformed = imputer.fit_transform(make_df(data_na_dob))
    assert imputer.n_features_in_ == 6
    assert imputer.variables_ == [
        "Name", "City", "Studies", "Age", "Marks", "dob"
    ]
    assert isinstance(X_transformed, make_df)
    assert X_transformed.shape == (5, 6)
    for var in imputer.variables_:
        assert null_count(X_transformed, var) == 0


def test_detect_variables_with_na_in_variables_entered_by_user(make_df, data_na_dob):
    imputer = DropMissingData(
        missing_only=True, variables=["City", "Studies", "Age", "dob"]
    )
    X_transformed = imputer.fit_transform(make_df(data_na_dob))
    assert imputer.variables == ["City", "Studies", "Age", "dob"]
    # dob never has NA in the train set, so it's dropped from variables_
    assert imputer.variables_ == ["City", "Studies", "Age"]
    assert isinstance(X_transformed, make_df)
    assert X_transformed.shape == (6, 6)
    assert to_dict(X_transformed)["Age"] == [20, 21, 23, 40, 41, 37]


def test_return_na_data_method(make_df, data_na_dob):
    df_na = make_df(data_na_dob)

    # test with vars and threshold: return_na_data must return the exact
    # complement of transform() - row 2 has 2 of 4 variables present, which
    # meets thresh=2 and is therefore *kept* by transform(), so it must NOT
    # also show up here.
    imputer = DropMissingData(
        threshold=0.5, variables=["City", "Studies", "Age", "Marks"]
    )
    imputer.fit_transform(df_na)
    X_nona = imputer.return_na_data(df_na)
    assert isinstance(X_nona, make_df)
    assert X_nona.shape[0] == 1
    assert to_dict(X_nona)["Age"] == [None]

    # test without vars & threshold
    imputer = DropMissingData()
    imputer.fit_transform(df_na)
    X_nona = imputer.return_na_data(df_na)
    assert isinstance(X_nona, make_df)
    assert X_nona.shape[0] == 3
    assert to_dict(X_nona)["Age"] == [19, None, 40]


def test_transform_and_return_na_data_partition_input(make_df, data_na_dob):
    # transform() (rows kept) and return_na_data() (rows dropped) must
    # partition the input exactly: no row in both, no row in neither.
    df_na = make_df(data_na_dob)
    for threshold in [None, 1, 0.75, 0.5, 0.25, 0.01]:
        imputer = DropMissingData(
            threshold=threshold, variables=["City", "Studies", "Age", "Marks"]
        )
        imputer.fit(df_na)
        kept = imputer.transform(df_na)
        dropped = imputer.return_na_data(df_na)
        assert kept.shape[0] + dropped.shape[0] == df_na.shape[0]
        kept_age = set(to_dict(kept)["Age"])
        dropped_age = set(to_dict(dropped)["Age"])
        assert kept_age.isdisjoint(dropped_age)


def test_error_when_missing_only_not_bool():
    with pytest.raises(ValueError):
        DropMissingData(missing_only="missing_only")


def test_threshold(make_df, data_na_dob):
    df_na = make_df(data_na_dob)

    # Each row must have 100% data available
    imputer = DropMissingData(threshold=1)
    X = imputer.fit_transform(df_na)
    assert isinstance(X, make_df)
    assert to_dict(X)["Age"] == [20, 21, 23, 41, 37]

    # Each row must have at least 1% data available
    imputer = DropMissingData(threshold=0.01)
    X = imputer.fit_transform(df_na)
    assert to_dict(X)["Age"] == [20, 21, 19, None, 23, 40, 41, 37]

    # Each row must have at least 50% data available
    imputer = DropMissingData(threshold=0.50)
    X = imputer.fit_transform(df_na)
    assert to_dict(X)["Age"] == [20, 21, 19, 23, 40, 41, 37]

    # threshold overrides missing_only, so the same 3 checks hold verbatim
    # with missing_only=False:
    imputer = DropMissingData(threshold=1, missing_only=False)
    X = imputer.fit_transform(df_na)
    assert to_dict(X)["Age"] == [20, 21, 23, 41, 37]

    imputer = DropMissingData(threshold=0.01, missing_only=False)
    X = imputer.fit_transform(df_na)
    assert to_dict(X)["Age"] == [20, 21, 19, None, 23, 40, 41, 37]

    imputer = DropMissingData(threshold=0.50, missing_only=False)
    X = imputer.fit_transform(df_na)
    assert to_dict(X)["Age"] == [20, 21, 19, 23, 40, 41, 37]


def test_threshold_value_error():
    with pytest.raises(ValueError):
        DropMissingData(threshold=1.01)

    with pytest.raises(ValueError):
        DropMissingData(threshold=-0.01)

    with pytest.raises(ValueError):
        DropMissingData(threshold=0)


def test_threshold_with_variables(make_df, data_na_dob):
    df_na = make_df(data_na_dob)

    # Each row must have 100% data available for column ['Marks']
    imputer = DropMissingData(threshold=1, variables=["Marks"])
    X = imputer.fit_transform(df_na)
    assert isinstance(X, make_df)
    assert to_dict(X)["Age"] == [20, 21, 19, 23, 41, 37]

    # Each row must have 75% data available for ['City', 'Studies', 'Age', 'Marks']
    imputer = DropMissingData(
        threshold=0.75, variables=["City", "Studies", "Age", "Marks"]
    )
    X = imputer.fit_transform(df_na)
    assert to_dict(X)["Age"] == [20, 21, 23, 40, 41, 37]


def test_missing_only_finds_no_variables_leaves_data_unchanged(make_df):
    # A clean training set has nothing for missing_only=True to select:
    # variables_ ends up empty, and transform()/return_na_data() must not
    # error on the narwhals horizontal-expression path with 0 columns.
    clean_data = {"x1": [1, 2, 3], "x2": [4, 5, 6]}
    X = make_df(clean_data)
    imputer = DropMissingData()
    Xt = imputer.fit_transform(X)
    assert imputer.variables_ == []
    assert isinstance(Xt, make_df)
    assert to_dict(Xt) == clean_data
    X_nona = imputer.return_na_data(X)
    assert isinstance(X_nona, make_df)
    assert X_nona.shape == (0, 2)
