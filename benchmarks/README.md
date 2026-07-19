# Feature-engine performance benchmarks

This directory contains a standalone benchmark suite that measures the
wall-clock time of `fit()` and `transform()` for a representative set of
feature-engine transformers.

It was created to establish a performance baseline **before** the version 2.0
refactoring (polars support and numpy-based computation), so that we can
verify that the refactoring makes the transformers faster, not slower.

The suite is deliberately self-contained: it needs no extra dependencies, it
is not collected by pytest, and it does not run in CI.

## Running the benchmarks

From the repository root:

```bash
python benchmarks/run_benchmarks.py
```

Useful options:

| option | default | meaning |
|---|---|---|
| `--rows N` | 100000 | number of rows in the synthetic datasets |
| `--repeats N` | 5 | timed repetitions per benchmark (min is reported) |
| `--filter STR` | | run only benchmarks whose name contains STR |
| `--save FILE` | | save the results as a JSON baseline |
| `--compare FILE` | | compare this run against a saved baseline |
| `--threshold F` | 0.10 | regression threshold for `--compare` (0.10 = 10%) |

## Workflow: checking that a refactor does not slow things down

1. On the branch **before** the change, save a baseline:

   ```bash
   python benchmarks/run_benchmarks.py --save benchmarks/baselines/before.json
   ```

2. Switch to the branch **with** the change, and compare:

   ```bash
   python benchmarks/run_benchmarks.py --compare benchmarks/baselines/before.json
   ```

   The run exits with code 1 and prints `<< REGRESSION` for every benchmark
   that is more than the threshold slower than the baseline.

Timings are machine-specific: only compare runs made on the same machine with
the same versions of python, pandas, numpy and scikit-learn (the versions are
stored in the baseline and printed during comparison).

## What is covered

One or more representative transformers per module: imputation, encoding,
discretisation, outliers, transformation, scaling, creation, datetime,
timeseries, selection, preprocessing and text.

Deliberately **not** covered: transformers whose runtime is dominated by
scikit-learn model training rather than dataframe operations (the decision
tree transformers/encoders and the model-based selectors such as
`SelectByShuffling` or `RecursiveFeatureElimination`). The 2.0 refactoring
targets the dataframe layer, which is noise compared to model training in
those transformers.

## Baselines

`baselines/` holds saved runs. The committed baseline
(`2026-07-19_macos_py3.14_pandas3.0.3.json`) is the pre-refactor reference
recorded on an Apple Silicon laptop; treat it as indicative only and record
your own baseline on your own machine before starting performance work.
