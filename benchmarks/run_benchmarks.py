"""Performance benchmarks for feature-engine transformers.

This suite measures the wall-clock time of fit() and transform() for a
representative set of transformers, on synthetic data of configurable size.
It exists to establish a performance baseline before the 2.0 refactoring
(polars support + numpy-based compute), and to verify afterwards that the
refactoring makes the transformers faster, not slower.

The suite is standalone on purpose: it needs no pytest plugins and it is not
collected by the regular test suite or CI.

Usage
-----
Run all benchmarks and print a table:

    python benchmarks/run_benchmarks.py

Save a baseline:

    python benchmarks/run_benchmarks.py --save benchmarks/baselines/my_run.json

Compare a new run against a saved baseline (exit code 1 if any benchmark is
slower than the baseline by more than --threshold, default 10%):

    python benchmarks/run_benchmarks.py --compare benchmarks/baselines/my_run.json

Other options: --rows N (default 100_000), --repeats N (default 5),
--filter SUBSTRING (run only matching benchmarks).

Note: timings are machine-specific. Compare runs only on the same machine and
the same versions of python and the dependencies.
"""

import argparse
import datetime
import gc
import json
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd
import sklearn

import feature_engine
from feature_engine.creation import CyclicalFeatures, MathFeatures, RelativeFeatures
from feature_engine.datetime import DatetimeFeatures, DatetimeSubtraction
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    GeometricWidthDiscretiser,
)
from feature_engine.encoding import (
    CountFrequencyEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
    StringSimilarityEncoder,
    WoEEncoder,
)
from feature_engine.imputation import (
    AddMissingIndicator,
    ArbitraryNumberImputer,
    CategoricalImputer,
    DropMissingData,
    EndTailImputer,
    MeanMedianImputer,
    RandomSampleImputer,
)
from feature_engine.outliers import OutlierTrimmer, Winsorizer
from feature_engine.preprocessing import MatchCategories, MatchVariables
from feature_engine.scaling import MeanNormalizationScaler
from feature_engine.selection import (
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
    DropHighPSIFeatures,
    SelectByInformationValue,
)
from feature_engine.text import TextFeatures
from feature_engine.timeseries.forecasting import (
    ExpandingWindowFeatures,
    LagFeatures,
    WindowFeatures,
)
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer

# --------------------------------------------------------------------------
# Synthetic datasets
# --------------------------------------------------------------------------

WORDS = [
    "feature", "engine", "pandas", "polars", "numpy", "python", "model",
    "variable", "transform", "encode", "impute", "select", "data", "frame",
]


def make_data(n_rows: int, seed: int = 42) -> dict:
    """Create the synthetic datasets used by the benchmarks."""
    rng = np.random.default_rng(seed)

    # strictly positive numerical data (so that log-type transformers work)
    num = {f"num_{i}": rng.lognormal(0.0, 1.0, n_rows) for i in range(10)}
    df_num = pd.DataFrame(num)

    # numerical data with 10% missing values
    df_num_na = df_num.copy()
    for col in df_num_na.columns[:5]:
        mask = rng.random(n_rows) < 0.10
        df_num_na.loc[mask, col] = np.nan

    # categorical data: 5 string variables with 20 categories each
    categories = [f"cat_{i}" for i in range(20)]
    cat = {
        f"str_{i}": rng.choice(categories, n_rows).astype(object)
        for i in range(5)
    }
    df_cat = pd.DataFrame(cat)

    df_cat_na = df_cat.copy()
    for col in df_cat_na.columns[:3]:
        mask = rng.random(n_rows) < 0.10
        df_cat_na.loc[mask, col] = np.nan

    # mixed data
    df_mixed = pd.concat([df_num.iloc[:, :5], df_cat.iloc[:, :3]], axis=1)
    df_mixed_na = pd.concat(
        [df_num_na.iloc[:, :5], df_cat_na.iloc[:, :3]], axis=1
    )

    # quasi-constant and duplicated variables for the selectors
    df_const = df_mixed.copy()
    df_const["const_1"] = 1.0
    quasi = np.ones(n_rows)
    quasi[: max(1, n_rows // 100)] = 0.0
    df_const["quasi_1"] = quasi

    df_dup = df_num.iloc[:, :5].copy()
    df_dup["dup_0"] = df_dup["num_0"]
    df_dup["dup_1"] = df_dup["num_1"]

    # correlated variables
    base = rng.normal(size=(n_rows, 10))
    corr = {f"corr_{i}": base[:, i] for i in range(10)}
    for i in range(10):
        corr[f"corr_noisy_{i}"] = base[:, i] + rng.normal(
            0.0, 0.1, n_rows
        )
    df_corr = pd.DataFrame(corr)

    # datetime data
    start = pd.Timestamp("2020-01-01")
    dts = {
        f"dt_{i}": pd.to_datetime(
            rng.integers(0, 4 * 365 * 24, n_rows), unit="h", origin=start
        )
        for i in range(3)
    }
    df_dt = pd.DataFrame(dts)
    df_dt_str = df_dt.astype(str)

    # time series data with a sorted datetime index
    df_ts = pd.DataFrame(
        {f"num_{i}": rng.normal(size=n_rows) for i in range(5)},
        index=pd.date_range("2020-01-01", periods=n_rows, freq="min"),
    )

    # text data
    n_words = rng.integers(3, 15, n_rows)
    df_text = pd.DataFrame(
        {"text": [" ".join(rng.choice(WORDS, k)) for k in n_words]}
    )

    y = pd.Series(rng.integers(0, 2, n_rows), name="target")

    return {
        "num": df_num,
        "num_na": df_num_na,
        "cat": df_cat,
        "cat_na": df_cat_na,
        "mixed": df_mixed,
        "mixed_na": df_mixed_na,
        "const": df_const,
        "dup": df_dup,
        "corr": df_corr,
        "dt": df_dt,
        "dt_str": df_dt_str,
        "ts": df_ts,
        "text": df_text,
        "y": y,
    }


# --------------------------------------------------------------------------
# Benchmark cases: (name, transformer factory, dataset key, needs_y)
# --------------------------------------------------------------------------

CASES = [
    # imputation
    ("MeanMedianImputer", lambda: MeanMedianImputer(), "num_na", False),
    ("EndTailImputer", lambda: EndTailImputer(), "num_na", False),
    ("ArbitraryNumberImputer", lambda: ArbitraryNumberImputer(), "num_na", False),
    (
        "CategoricalImputer(frequent)",
        lambda: CategoricalImputer(imputation_method="frequent"),
        "cat_na",
        False,
    ),
    ("AddMissingIndicator", lambda: AddMissingIndicator(), "mixed_na", False),
    ("DropMissingData", lambda: DropMissingData(), "mixed_na", False),
    (
        "RandomSampleImputer(general)",
        lambda: RandomSampleImputer(random_state=0),
        "mixed_na",
        False,
    ),
    # encoding
    ("CountFrequencyEncoder", lambda: CountFrequencyEncoder(), "cat", False),
    (
        "OrdinalEncoder(ordered)",
        lambda: OrdinalEncoder(encoding_method="ordered"),
        "cat",
        True,
    ),
    ("MeanEncoder", lambda: MeanEncoder(), "cat", True),
    ("WoEEncoder", lambda: WoEEncoder(), "cat", True),
    (
        "RareLabelEncoder",
        lambda: RareLabelEncoder(tol=0.01, n_categories=2),
        "cat",
        False,
    ),
    ("OneHotEncoder(top10)", lambda: OneHotEncoder(top_categories=10), "cat", False),
    (
        "StringSimilarityEncoder(top5)",
        lambda: StringSimilarityEncoder(top_categories=5),
        "cat",
        False,
    ),
    # discretisation
    ("EqualFrequencyDiscretiser", lambda: EqualFrequencyDiscretiser(), "num", False),
    ("EqualWidthDiscretiser", lambda: EqualWidthDiscretiser(), "num", False),
    ("GeometricWidthDiscretiser", lambda: GeometricWidthDiscretiser(), "num", False),
    # outliers
    ("Winsorizer(gaussian)", lambda: Winsorizer(), "num", False),
    (
        "Winsorizer(quantiles)",
        lambda: Winsorizer(capping_method="quantiles", fold=0.05),
        "num",
        False,
    ),
    (
        "OutlierTrimmer(iqr)",
        lambda: OutlierTrimmer(capping_method="iqr", tail="both"),
        "num",
        False,
    ),
    # transformation
    ("LogTransformer", lambda: LogTransformer(), "num", False),
    (
        "YeoJohnsonTransformer(3vars)",
        lambda: YeoJohnsonTransformer(variables=["num_0", "num_1", "num_2"]),
        "num",
        False,
    ),
    # scaling
    ("MeanNormalizationScaler", lambda: MeanNormalizationScaler(), "num", False),
    # creation
    ("CyclicalFeatures", lambda: CyclicalFeatures(), "num", False),
    (
        "MathFeatures(sum,mean,std)",
        lambda: MathFeatures(
            variables=[f"num_{i}" for i in range(5)],
            func=["sum", "mean", "std"],
        ),
        "num",
        False,
    ),
    (
        "RelativeFeatures(add,sub)",
        lambda: RelativeFeatures(
            variables=["num_0", "num_1", "num_2"],
            reference=["num_3"],
            func=["add", "sub"],
        ),
        "num",
        False,
    ),
    # datetime
    ("DatetimeFeatures(datetime)", lambda: DatetimeFeatures(), "dt", False),
    ("DatetimeFeatures(strings)", lambda: DatetimeFeatures(), "dt_str", False),
    (
        "DatetimeSubtraction",
        lambda: DatetimeSubtraction(variables=["dt_0"], reference=["dt_1"]),
        "dt",
        False,
    ),
    # time series
    ("LagFeatures", lambda: LagFeatures(periods=[1, 2, 3]), "ts", False),
    (
        "WindowFeatures",
        lambda: WindowFeatures(window=[3, 7], functions=["mean", "std"]),
        "ts",
        False,
    ),
    (
        "ExpandingWindowFeatures",
        lambda: ExpandingWindowFeatures(functions=["mean"]),
        "ts",
        False,
    ),
    # selection (dataframe-bound selectors; the model-based selectors are
    # dominated by scikit-learn compute and are not part of this suite)
    (
        "DropConstantFeatures",
        lambda: DropConstantFeatures(tol=0.98),
        "const",
        False,
    ),
    ("DropDuplicateFeatures", lambda: DropDuplicateFeatures(), "dup", False),
    (
        "DropCorrelatedFeatures",
        lambda: DropCorrelatedFeatures(threshold=0.8),
        "corr",
        False,
    ),
    ("DropHighPSIFeatures", lambda: DropHighPSIFeatures(bins=5), "num", False),
    (
        "SelectByInformationValue",
        lambda: SelectByInformationValue(bins=5),
        "cat",
        True,
    ),
    # preprocessing
    ("MatchCategories", lambda: MatchCategories(), "cat", False),
    ("MatchVariables", lambda: MatchVariables(), "mixed", False),
    # text
    (
        "TextFeatures(3features)",
        lambda: TextFeatures(
            variables="text",
            features=["word_count", "char_count", "lexical_diversity"],
        ),
        "text",
        False,
    ),
]


# --------------------------------------------------------------------------
# Timing machinery
# --------------------------------------------------------------------------

def _time_call(func, repeats: int) -> list:
    """Time func() repeats times (plus one timed warmup), in seconds.

    To keep the total runtime of the suite bounded, slow calls are repeated
    fewer times: calls slower than 2s run once more, calls slower than 0.5s
    run at most twice more. The warmup timing is discarded unless it is the
    only measurement available.
    """
    gc.collect()
    start = time.perf_counter()
    func()  # warmup
    warmup = time.perf_counter() - start

    if warmup > 2.0:
        repeats = 1
    elif warmup > 0.5:
        repeats = min(repeats, 2)

    timings = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return timings


def run_benchmarks(data: dict, repeats: int, name_filter: str = "") -> dict:
    results = {}
    y = data["y"]
    for name, factory, data_key, needs_y in CASES:
        if name_filter and name_filter.lower() not in name.lower():
            continue
        X = data[data_key]
        target = y if needs_y else None

        fit_times = _time_call(lambda: factory().fit(X, target), repeats)

        fitted = factory().fit(X, target)
        transform_times = _time_call(lambda: fitted.transform(X), repeats)

        results[name] = {
            "fit": {
                "min": min(fit_times),
                "median": statistics.median(fit_times),
            },
            "transform": {
                "min": min(transform_times),
                "median": statistics.median(transform_times),
            },
        }
        print(
            f"{name:32s} fit {min(fit_times) * 1000:10.1f} ms   "
            f"transform {min(transform_times) * 1000:10.1f} ms"
        )
    return results


def collect_meta(n_rows: int, repeats: int) -> dict:
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "feature_engine": feature_engine.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "n_rows": n_rows,
        "repeats": repeats,
    }


def compare(results: dict, baseline_path: str, threshold: float) -> int:
    with open(baseline_path) as f:
        baseline = json.load(f)

    print()
    print(f"Comparing against: {baseline_path}")
    print(f"Baseline meta: {baseline['meta']}")
    print()
    header = (
        f"{'benchmark':32s} {'step':10s} {'baseline':>10s} "
        f"{'current':>10s} {'ratio':>7s}"
    )
    print(header)
    print("-" * len(header))

    regressions = []
    for name, res in results.items():
        base = baseline["results"].get(name)
        if base is None:
            print(f"{name:32s} (not in baseline)")
            continue
        for step in ("fit", "transform"):
            old = base[step]["min"]
            new = res[step]["min"]
            ratio = new / old if old > 0 else float("inf")
            flag = ""
            if ratio > 1 + threshold:
                flag = "  << REGRESSION"
                regressions.append((name, step, ratio))
            elif ratio < 1 - threshold:
                flag = "  (faster)"
            print(
                f"{name:32s} {step:10s} {old * 1000:8.1f}ms "
                f"{new * 1000:8.1f}ms {ratio:6.2f}x{flag}"
            )

    print()
    if regressions:
        print(f"{len(regressions)} REGRESSION(S) above {threshold:.0%}:")
        for name, step, ratio in regressions:
            print(f"  - {name} / {step}: {ratio:.2f}x slower")
        return 1
    print(f"No regressions above {threshold:.0%}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--filter", type=str, default="")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--compare", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()

    meta = collect_meta(args.rows, args.repeats)
    print("Environment:", json.dumps(meta, indent=2))
    print(f"\nGenerating synthetic data with {args.rows} rows...")
    data = make_data(args.rows)
    print("Running benchmarks...\n")

    results = run_benchmarks(data, args.repeats, args.filter)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"\nSaved baseline to {args.save}")

    if args.compare:
        return compare(results, args.compare, args.threshold)

    return 0


if __name__ == "__main__":
    sys.exit(main())
