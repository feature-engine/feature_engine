import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier

from feature_engine.selection import ProbeFeaturesSelection


def test_generate_probe_features(df_test):

    X, y = df_test

    # instantiate classifier
    clsfr = RandomForestClassifier()
    clsfr.fit(X, y)

    # instantiate selector
    sel = ProbeFeaturesSelection(
        estimator=clsfr,
        scoring="roc_auc",
        n_iter=5,
        seed=0,
        confirm_variables=False
    )
    sel.fit(X, y)
    importances = sel.probe_features_

    assert importances.shape == (X.shape[0], 3)