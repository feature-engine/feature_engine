"""Benchmarks for the feature selection transformers.

Selectors do most of their work in ``fit``, so these benchmarks focus on it.
The estimator based selectors run on the smaller dataframe and with a light
estimator and 2 folds to keep the runtime reasonable.
"""

import pytest
from sklearn.tree import DecisionTreeClassifier

from feature_engine.selection import (
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
    DropFeatures,
    DropHighPSIFeatures,
    SelectByInformationValue,
    SelectBySingleFeaturePerformance,
    SelectByTargetEncoding,
    SmartCorrelatedSelection,
)

from .conftest import categorical_vars, numerical_vars

NUM_VARS = numerical_vars()
CAT_VARS = categorical_vars()


def _estimator():
    return DecisionTreeClassifier(max_depth=3, random_state=0)


def test_drop_features_transform(benchmark, df_big):
    selector = DropFeatures(features_to_drop=NUM_VARS[:3])
    selector.fit(df_big)
    benchmark(selector.transform, df_big)


def test_drop_constant_features_fit(benchmark, df_big):
    selector = DropConstantFeatures(tol=0.998)
    benchmark(selector.fit, df_big)


def test_drop_duplicate_features_fit(benchmark, df_small):
    # Compares every pair of columns, so it runs on the smaller dataframe.
    selector = DropDuplicateFeatures()
    benchmark(selector.fit, df_small)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_drop_correlated_features_fit(benchmark, df_big, method):
    selector = DropCorrelatedFeatures(variables=NUM_VARS, method=method, threshold=0.8)
    benchmark(selector.fit, df_big)


def test_smart_correlated_selection_fit(benchmark, df_big, y_binary_big):
    selector = SmartCorrelatedSelection(
        variables=NUM_VARS,
        selection_method="variance",
        threshold=0.8,
    )
    benchmark(selector.fit, df_big, y_binary_big)


def test_drop_high_psi_features_fit(benchmark, df_big):
    selector = DropHighPSIFeatures(variables=NUM_VARS, bins=10, split_frac=0.5)
    benchmark(selector.fit, df_big)


def test_select_by_information_value_fit(benchmark, df_big, y_binary_big):
    selector = SelectByInformationValue(variables=CAT_VARS, threshold=0.2)
    benchmark(selector.fit, df_big, y_binary_big)


def test_select_by_target_encoding_fit(benchmark, df_small, y_binary):
    selector = SelectByTargetEncoding(
        variables=NUM_VARS[:4] + CAT_VARS[:2],
        bins=5,
        cv=2,
        scoring="roc_auc",
        regression=False,
    )
    benchmark(selector.fit, df_small, y_binary)


def test_select_by_single_feature_performance_fit(benchmark, df_small, y_binary):
    selector = SelectBySingleFeaturePerformance(
        estimator=_estimator(),
        variables=NUM_VARS,
        scoring="roc_auc",
        cv=2,
    )
    benchmark(selector.fit, df_small, y_binary)
