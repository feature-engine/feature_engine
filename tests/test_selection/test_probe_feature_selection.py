import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier

from feature_engine.selection import ProbeFeatureSelection


def test_generate_probe_feature(df_test):

    X, y = df_test

    # instantiate classifier
    clsfr = RandomForestClassifier()
    clsfr.fit(X, y)

    # instantiate selector
    sel = ProbeFeatureSelection(
        estimator=clsfr,
        scoring="roc_auc",
        distribution="normal",
        cut_rate=0.2,
        cv=5,
        n_iter=5,
        random_state=0,
        confirm_variables=False
    )
    sel.fit(X, y)
    probe_feature_data = sel.probe_feature_data_

    # expected results
    expected_results_head = {
        "rndm_gaussian_var": [5.292, 1.200, 2.936, 6.723, 5.603],
    }
    expected_results_head_df = pd.DataFrame(expected_results_head)

    expected_results_tail = {
        "rndm_gaussian_var": [1.239, -0.595, 0.283, -3.443, -1.074],
    }
    expected_results_tail_df = pd.DataFrame(
        data=expected_results_tail,
        index=range(995, 1000),
    )

    assert probe_feature_data.shape == (X.shape[0], 1)
    assert probe_feature_data.head().round(3).equals(expected_results_head_df)
    assert probe_feature_data.tail().round(3).equals(expected_results_tail_df)