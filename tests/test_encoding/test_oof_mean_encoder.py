import numpy as np
import pandas as pd
import pytest
from numpy import nan
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from feature_engine.encoding import MeanEncoder, OOFMeanEncoder


@pytest.fixture
def df_enc():
    return pd.DataFrame(
        {
            "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
            "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
            "target": [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        }
    )


@pytest.fixture
def df_high_cardinality():
    """Every category has a single observation, so the in-sample target mean of a
    category equals the target of its own row: maximum leakage."""
    n = 200
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "cat": [f"c{i}" for i in range(n)],
            "num": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )


# ---------------------------------------------------------------------------
# init params
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("params", [("raise", True, "auto"), ("ignore", False, 1)])
def test_init_param_assignment(params):
    OOFMeanEncoder(
        missing_values=params[0],
        ignore_format=params[1],
        unseen=params[0],
        smoothing=params[2],
    )


@pytest.mark.parametrize(
    "errors", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_gets_not_permitted_value(errors):
    with pytest.raises(ValueError):
        OOFMeanEncoder(unseen=errors)


@pytest.mark.parametrize("smoothing", ["hello", ["auto"], -1])
def test_raises_error_when_not_allowed_smoothing_param_in_init(smoothing):
    with pytest.raises(ValueError):
        OOFMeanEncoder(smoothing=smoothing)


@pytest.mark.parametrize("cv", [1, 0, -3, 2.5, "5"])
def test_raises_error_when_cv_is_not_an_integer_greater_than_1(cv):
    with pytest.raises(ValueError):
        OOFMeanEncoder(cv=cv)


@pytest.mark.parametrize("shuffle", ["yes", 1, None])
def test_raises_error_when_shuffle_is_not_boolean(shuffle):
    with pytest.raises(ValueError):
        OOFMeanEncoder(shuffle=shuffle)


@pytest.mark.parametrize("stratify", ["yes", 1, "auto"])
def test_raises_error_when_stratify_is_not_boolean_or_none(stratify):
    with pytest.raises(ValueError):
        OOFMeanEncoder(stratify=stratify)


def test_get_params_and_clone():
    encoder = OOFMeanEncoder(variables=["var_A"], cv=3, shuffle=True, random_state=3)
    params = encoder.get_params()
    assert params["cv"] == 3
    assert params["shuffle"] is True
    assert params["random_state"] == 3
    assert isinstance(clone(encoder), OOFMeanEncoder)


# ---------------------------------------------------------------------------
# fit / transform: parity with MeanEncoder
# ---------------------------------------------------------------------------
def test_fit_learns_same_mapping_as_mean_encoder(df_enc):
    oof = OOFMeanEncoder(variables=["var_A"], cv=4).fit(
        df_enc[["var_A", "var_B"]], df_enc["target"]
    )
    mean = MeanEncoder(variables=["var_A"]).fit(
        df_enc[["var_A", "var_B"]], df_enc["target"]
    )

    assert oof.encoder_dict_ == mean.encoder_dict_
    # sanity: the mapping is the full-data mean per category, and y_prior_ is the
    # overall target mean (not a per category mean).
    assert oof.encoder_dict_["var_A"]["A"] == pytest.approx(5 / 6)
    assert oof.y_prior_ == pytest.approx(df_enc["target"].mean())
    assert oof.y_prior_ != oof.encoder_dict_["var_A"]["A"]


def test_transform_matches_mean_encoder(df_enc):
    oof = OOFMeanEncoder(variables=["var_A"], cv=4).fit(
        df_enc[["var_A", "var_B"]], df_enc["target"]
    )
    mean = MeanEncoder(variables=["var_A"]).fit(
        df_enc[["var_A", "var_B"]], df_enc["target"]
    )

    pd.testing.assert_frame_equal(
        oof.transform(df_enc[["var_A", "var_B"]]),
        mean.transform(df_enc[["var_A", "var_B"]]),
    )


def test_error_when_transforming_before_fitting(df_enc):
    with pytest.raises(NotFittedError):
        OOFMeanEncoder().transform(df_enc[["var_A", "var_B"]])


# ---------------------------------------------------------------------------
# the point of the transformer: no leakage in fit_transform
# ---------------------------------------------------------------------------
def test_fit_transform_does_not_leak_the_target(df_high_cardinality):
    X, y = df_high_cardinality[["cat", "num"]], df_high_cardinality["target"]

    # in-sample mean encoding: each category is encoded with its own row's target.
    in_sample = MeanEncoder(variables=["cat"]).fit(X, y).transform(X)["cat"]
    assert np.corrcoef(in_sample, y)[0, 1] > 0.99

    # out-of-fold encoding: a row's own target is never used to encode it.
    oof = OOFMeanEncoder(variables=["cat"], cv=5, shuffle=True, random_state=0)
    X_oof = oof.fit_transform(X, y)["cat"]

    assert X_oof.isnull().sum() == 0
    assert abs(np.corrcoef(X_oof, y)[0, 1]) < 0.2


def test_oof_values_are_fold_dependent_unlike_in_sample_encoding(df_enc):
    """The in-sample encoding is a deterministic function of the category — one single
    value per category, which is exactly what leaks the target. The out-of-fold
    encoding is not: every fold learns its own mapping, so two rows of the same
    category can receive different values."""
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]

    in_sample = MeanEncoder(variables=["var_A"]).fit(X, y).transform(X)["var_A"]
    encoder = OOFMeanEncoder(variables=["var_A"], cv=3, shuffle=True, random_state=1)
    X_oof = encoder.fit_transform(X, y)["var_A"]

    assert in_sample.groupby(X["var_A"]).nunique().max() == 1
    assert in_sample.nunique() == X["var_A"].nunique()

    # every category is mapped to more than one value across folds
    assert X_oof.groupby(X["var_A"]).nunique().min() > 1
    assert X_oof.nunique() > X["var_A"].nunique()

    # and the values remain probability-like
    assert 0 <= X_oof.min() and X_oof.max() <= 1


def test_fit_transform_leaves_encoder_fitted_for_new_data(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]

    oof = OOFMeanEncoder(variables=["var_A"], cv=4)
    oof.fit_transform(X, y)

    assert oof.n_splits_ == 4
    assert len(oof.fold_mappings_) == 4
    assert len(oof.fold_priors_) == 4
    assert oof.variables_ == ["var_A"]
    assert oof.n_features_in_ == 2
    assert oof.feature_names_in_ == ["var_A", "var_B"]

    # transform on new data uses the mapping learnt on the whole data set
    new = pd.DataFrame({"var_A": ["A", "B", "C"], "var_B": ["A", "B", "C"]})
    transformed = oof.transform(new)
    assert transformed["var_A"].tolist() == [
        oof.encoder_dict_["var_A"]["A"],
        oof.encoder_dict_["var_A"]["B"],
        oof.encoder_dict_["var_A"]["C"],
    ]


def test_fit_transform_preserves_shape_columns_and_untouched_variables(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]

    X_oof = OOFMeanEncoder(variables=["var_A"], cv=2).fit_transform(X, y)

    assert X_oof.shape == X.shape
    assert X_oof.columns.tolist() == X.columns.tolist()
    pd.testing.assert_series_equal(X_oof["var_B"], X["var_B"])
    assert X_oof["var_A"].dtype == "float64"


def test_fit_transform_works_with_categorical_dtype_and_numerical_variables():
    df = pd.DataFrame(
        {
            "cat": pd.Categorical(["a", "b", "c", "a", "b", "c"]),
            "num": [1, 1, 2, 2, 3, 3],
            "target": [0, 1, 1, 0, 1, 1],
        }
    )

    X_oof = OOFMeanEncoder(
        variables=["cat", "num"], ignore_format=True, cv=2
    ).fit_transform(df[["cat", "num"]], df["target"])

    assert X_oof["cat"].dtype == "float64"
    assert X_oof["num"].dtype == "float64"
    assert X_oof.isnull().sum().sum() == 0


def test_oof_values_approximate_full_data_mapping(df_enc):
    """The out-of-fold means are estimated from the same data, so per original
    category they should be in the same ballpark as the full-data mean."""
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]

    oof = OOFMeanEncoder(variables=["var_A"], cv=10, shuffle=True, random_state=2)
    X_oof = oof.fit_transform(X, y)
    X_oof["category"] = X["var_A"].to_numpy()

    oof_means = X_oof.groupby("category")["var_A"].mean()
    full_means = pd.Series(oof.encoder_dict_["var_A"])

    for category in ["A", "B", "C"]:
        assert abs(oof_means[category] - full_means[category]) < 0.35


# ---------------------------------------------------------------------------
# folds, reproducibility, unseen categories
# ---------------------------------------------------------------------------
def test_stratified_folds_are_used_for_categorical_targets(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]

    encoder = OOFMeanEncoder(variables=["var_A"], cv=4)
    encoder.fit_transform(X, y)

    # every fold keeps the share of positives close to the overall share
    overall = y.mean()
    assert encoder.n_splits_ == 4
    for mapping, prior in zip(encoder.fold_mappings_, encoder.fold_priors_):
        assert mapping["var_A"]
        assert abs(prior - overall) < 0.3


def test_plain_kfold_is_used_for_continuous_targets():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"cat": ["a", "b", "c", "d"] * 25})
    y = pd.Series(rng.normal(size=100))

    encoder = OOFMeanEncoder(variables=["cat"], cv=3)
    X_oof = encoder.fit_transform(X, y)

    assert encoder.n_splits_ == 3
    assert X_oof.isnull().sum().sum() == 0


def test_warns_and_falls_back_to_kfold_when_classes_are_too_small():
    X = pd.DataFrame({"cat": ["a", "b"] * 5})
    y = pd.Series([1] + [0] * 9)

    encoder = OOFMeanEncoder(variables=["cat"], cv=5)
    with pytest.warns(UserWarning):
        encoder.fit_transform(X, y)

    assert encoder.n_splits_ == 5


def test_fit_transform_is_reproducible_with_random_state(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]

    first = OOFMeanEncoder(
        variables=["var_A"], cv=3, shuffle=True, random_state=7
    ).fit_transform(X, y)
    second = OOFMeanEncoder(
        variables=["var_A"], cv=3, shuffle=True, random_state=7
    ).fit_transform(X, y)

    pd.testing.assert_frame_equal(first, second)


def test_fold_unseen_categories_are_encoded_with_the_fold_prior():
    """A category that only exists in the validation fold cannot be mapped, so it is
    encoded with the prior of the remaining folds instead of with NaN."""
    X = pd.DataFrame({"cat": ["a", "a", "b", "b", "c", "d"]})
    y = pd.Series([0, 1, 0, 1, 1, 0])

    encoder = OOFMeanEncoder(variables=["cat"], cv=2, stratify=False)
    X_oof = encoder.fit_transform(X, y)

    # fold 0 trains on rows 3-5 (b, c, d) and holds out rows 0-2 (a, a, b); 'a' is
    # therefore unseen and falls back to the prior of rows 3-5. Fold 1 does the same
    # for 'c' and 'd', which only appear in its held out rows.
    assert encoder.fold_priors_ == pytest.approx([2 / 3, 1 / 3])
    assert X_oof["cat"].tolist() == pytest.approx(
        [2 / 3, 2 / 3, 1.0, 0.0, 1 / 3, 1 / 3]
    )
    assert X_oof.isnull().sum().sum() == 0


def test_unseen_encode_uses_the_prior(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]
    oof = OOFMeanEncoder(variables=["var_A"], unseen="encode", cv=2).fit(X, y)

    new = pd.DataFrame({"var_A": ["D"], "var_B": ["A"]})
    assert oof.transform(new)["var_A"].iloc[0] == pytest.approx(y.mean())


def test_unseen_ignore_introduces_nan_in_transform(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]
    oof = OOFMeanEncoder(variables=["var_A"], unseen="ignore", cv=2).fit(X, y)

    new = pd.DataFrame({"var_A": ["D"], "var_B": ["A"]})
    assert np.isnan(oof.transform(new)["var_A"].iloc[0])


def test_unseen_raise_raises_in_fit_transform_when_category_is_missing_in_folds():
    X = pd.DataFrame({"cat": ["a", "a", "b", "b", "c", "d"]})
    y = pd.Series([0, 1, 0, 1, 1, 0])

    encoder = OOFMeanEncoder(variables=["cat"], cv=2, unseen="raise", stratify=False)

    with pytest.raises(ValueError):
        encoder.fit_transform(X, y)


# ---------------------------------------------------------------------------
# smoothing, missing values, inverse transform
# ---------------------------------------------------------------------------
def test_smoothing_shrinks_oof_values_towards_the_prior(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]

    no_smoothing = OOFMeanEncoder(
        variables=["var_A"], cv=2, shuffle=True, random_state=0
    ).fit_transform(X, y)["var_A"]
    smoothed = OOFMeanEncoder(
        variables=["var_A"], cv=2, shuffle=True, random_state=0, smoothing=10
    ).fit_transform(X, y)["var_A"]

    prior = y.mean()
    assert (smoothed - prior).abs().mean() <= (no_smoothing - prior).abs().mean()


def test_smoothing_auto_runs(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]

    oof = OOFMeanEncoder(variables=["var_A"], cv=2, smoothing="auto")
    X_oof = oof.fit_transform(X, y)

    assert X_oof["var_A"].isnull().sum() == 0


def test_missing_values_raise_by_default():
    X = pd.DataFrame({"cat": ["a", "b", nan, "b"]})
    y = pd.Series([0, 1, 1, 0])

    with pytest.raises(ValueError):
        OOFMeanEncoder(variables=["cat"], cv=2).fit_transform(X, y)


def test_missing_values_ignore(df_enc):
    X = df_enc[["var_A", "var_B"]].copy()
    X.loc[0, "var_A"] = nan
    y = df_enc["target"]

    X_oof = OOFMeanEncoder(
        variables=["var_A"], missing_values="ignore", cv=2
    ).fit_transform(X, y)

    assert X_oof["var_A"].isnull().sum() == 0


def test_inverse_transform_recovers_categories(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]

    oof = OOFMeanEncoder(variables=["var_A"], cv=2).fit(X, y)
    Xt = oof.transform(X)

    pd.testing.assert_frame_equal(oof.inverse_transform(Xt), X)


def test_inverse_transform_not_implemented_when_unseen_encode(df_enc):
    X, y = df_enc[["var_A", "var_B"]], df_enc["target"]
    oof = OOFMeanEncoder(variables=["var_A"], unseen="encode", cv=2).fit(X, y)

    with pytest.raises(NotImplementedError):
        oof.inverse_transform(oof.transform(X))


# ---------------------------------------------------------------------------
# end to end: leakage hurts a downstream model, out-of-fold encoding does not
# ---------------------------------------------------------------------------
def test_downstream_model_generalises_better_with_oof_encoding():
    rng = np.random.default_rng(42)
    n_categories = 300
    category_effect = rng.normal(size=n_categories)

    def make(n):
        idx = rng.integers(0, n_categories, size=n)
        cat = [f"c{i}" for i in idx]
        # the target depends on the category, with a lot of noise
        p = 1 / (1 + np.exp(-category_effect[idx]))
        y = rng.binomial(1, p)
        return pd.DataFrame({"cat": cat}), pd.Series(y)

    X_train, y_train = make(3000)
    X_valid, y_valid = make(3000)

    results = {}
    for name, encoder in [
        ("mean (in sample)", MeanEncoder(variables=["cat"])),
        ("oof", OOFMeanEncoder(variables=["cat"], cv=5, shuffle=True, random_state=0)),
    ]:
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_valid_enc = encoder.transform(X_valid)

        model = LogisticRegression().fit(X_train_enc, y_train)
        results[name] = roc_auc_score(y_valid, model.predict_proba(X_valid_enc)[:, 1])

    # the out-of-fold encoding is noisy but honest; the in-sample encoding is
    # over-confident and generalises no better (in practice it is worse).
    assert results["oof"] >= results["mean (in sample)"]
