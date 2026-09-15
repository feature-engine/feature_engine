# CLAUDE.md

General conventions live in AGENTS.md, imported below. This file adds the conventions
agreed while migrating transformers to narwhals.

@AGENTS.md

## Transformer code

- `check_X` and `check_X_y` return a narwhals dataframe. Bind it as `nw_X = check_X(X)`,
  pass the native `X` to the variable, NaN and feature-name helpers, compute on `nw_X`,
  and return `.to_native()`.
- When pandas keeps a native fast path, branch with
  `if nwd.is_pandas_dataframe(X):`, comment it with `# pandas is faster than narwhals.`
  and put the narwhals code in `else`.
- To use the target together with `X`, call `add_target_to_X(nw_X, y)` from
  `feature_engine/encoding/_helper_functions.py` and read it with `TARGET_NAME`. It works for
  series, list and array targets, and with pandas the column takes the index of `X`.
  Don't write this pairing again in a transformer.
- In the narwhals path, prefer narwhals expressions over Python loops on grouped results:
  aggregate with simple aggregations, then combine the columns in a `select`.
- Name temporary columns with double underscores (`__mean__`, `__count__`) so they can't
  clash with the user's columns.
- Don't add `# type: ignore`. If mypy complains because a parameter typed `Optional` is
  reassigned, store the value under a new name instead (for example `y_pd`).
- Don't add deprecation warnings. Behaviour changes go in directly, and a parameter that
  no longer applies is removed from the class and its docstring.
- When behaviour changes, check that the outputs shown in the user guide `.rst` examples
  are still correct.

## Init parameters

- Validate parameters in `__init__` only. Don't check them again in `fit` or `transform`,
  and don't test for errors raised by changing an attribute after init.
- For parameters that take a set of strings, check the type before the membership test,
  so lists, tuples, `None` and numbers raise the same error:

  ```python
  if not isinstance(encoding_method, str) or encoding_method not in [
      "ordered",
      "arbitrary",
  ]:
  ```

- Error messages follow the scikit-learn convention and end with
  `f"Got {param} instead."`.

## Tests

Every test file of a migrated transformer has the same structure, so they are easy to
maintain.

```python
# init parameters
def test_error_if_<param>_not_allowed(...)   # one test per error message
def test_init_param_assignment(...)          # several valid value combinations

# fit and transform
...
```

- Init error tests are parametrized with wrong values and wrong types.
- `test_init_param_assignment` checks every init parameter except `variables` and
  `return_empty`, which are tested elsewhere.
- Fit and transform tests don't assert init parameters.
- Every `pytest.raises` and `pytest.warns` matches the full message with
  `match=re.escape(msg)`, including `NotFittedError` and messages that come from
  scikit-learn. Matching the full message catches tests that pass for the wrong reason.

Backends and data:

- Request the `make_df` fixture from `tests/conftest.py`. Tests run once with pandas and
  once with polars.
- Use the helpers in `tests/backend_helpers.py`: `frame_to_dict`, `null_count` and
  `make_series`. Don't add per-file helpers that do the same.
- Data used by several test files of a module lives in that module's `conftest.py`, as
  fixtures that return plain dicts, with `None` for missing values. Data used by one file
  stays in that file.
- Pass the target as a series built with `make_series`, and add one test with the target
  as a list and as a numpy array.
- Check outputs with `assert isinstance(Xt, make_df)` and compare `frame_to_dict(Xt)` with
  a dict. Compare floats with `pytest.approx`.
- Backend-specific behaviour (category dtype, integer column names) keeps pandas-only
  tests with `pd.testing.assert_frame_equal`.
- Don't call polars' `to_pandas()` in tests: pyarrow is not installed locally or in CI.
- Name helpers after what they return (`frame_to_dict`, not `_cols`).

## Comments

Comments answer what the reader needs to know about the code: one line, two at most, in
source code and tests.

## Before pushing

- Run the tests of the changed transformer, `flake8 feature_engine tests` (lines of 88
  characters at most) and `mypy feature_engine`. Running mypy on single files ignores the
  exclusions in `pyproject.toml`.
- `narwhals-migration` has known failing tests. Compare the failing tests of the module
  with the target branch instead of expecting a clean run.

## Pull requests

- narwhals work targets the `narwhals-migration` branch.
- When a PR is built on another open PR and that one is squash-merged, rebase with
  `git rebase --onto origin/narwhals-migration <old base tip>` so the PR shows only its
  own files. Push with `--force-with-lease`.
- Don't end PR descriptions with the "Generated with Claude Code" line.
