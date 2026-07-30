# Benchmarks

This folder contains the performance benchmarks of Feature-engine. They are
written with [pytest-codspeed](https://github.com/CodSpeedHQ/pytest-codspeed)
and run on every push and pull request by the `CodSpeed` GitHub Actions
workflow, which reports the results to
[CodSpeed](https://app.codspeed.io/feature-engine/feature_engine).

## What is covered

One module per transformer family, benchmarking `fit` and `transform`
separately, since they have very different performance profiles:

| File                       | Covers                                                        |
| -------------------------- | ------------------------------------------------------------- |
| `test_imputation.py`       | Missing data imputers                                         |
| `test_encoding.py`         | Categorical encoders                                          |
| `test_discretisation.py`   | Discretisers                                                  |
| `test_outliers.py`         | Outlier cappers and trimmers                                  |
| `test_transformation.py`   | Mathematical transformers and scalers                         |
| `test_creation.py`         | Feature creation transformers                                 |
| `test_datetime.py`         | Datetime feature extraction                                   |
| `test_timeseries.py`       | Lag, window and expanding window features                     |
| `test_selection.py`        | Feature selectors                                             |
| `test_variable_handling.py`| Variable handling helpers, called by every transformer's `fit` |
| `test_pipeline.py`         | End to end pipelines and the preprocessing transformers       |

The data is synthetic and built in `conftest.py` fixtures, so data generation is
never part of what is measured. Dataframes are session scoped and shared by all
benchmarks.

## Running them locally

```bash
pip install -e .
pip install pytest pytest-codspeed

# quick check that the benchmarks run, with walltime measurements
pytest benchmarks/ --codspeed

# same measurements as CI, requires the CodSpeed CLI
codspeed run --mode simulation -- pytest benchmarks/ --codspeed
```

Running a single file or benchmark works as with any other pytest test:

```bash
pytest benchmarks/test_encoding.py --codspeed
pytest benchmarks/test_encoding.py::test_woe_encoder_fit --codspeed
```

## Adding a benchmark

- Reuse the dataframe fixtures from `conftest.py`. Use `df_big` for the
  vectorised transformers, `df_small` for the ones that train models
  (decision trees, cross-validation) and `df_tiny` for the row-wise ones.
- Do the `fit` outside of the measured section when benchmarking `transform`.
- Keep a single benchmark in the millisecond range: the whole suite runs under
  CPU simulation in CI, which is roughly two orders of magnitude slower than a
  plain run.
