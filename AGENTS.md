# AGENTS.md

Conventions for working in this repo. Optimize for speed, readability and
simplicity, in that order of how you decide: never ship a slow default when a
fast one is available.

## Inputs

Feature-engine transformers take dataframes (pandas, polars, or any other
narwhals-supported backend) as input, not numpy arrays. Don't add
handling for array input.

## Never import pandas in library code

pandas is an optional dependency (see `pyproject.toml` — it lives under
`[project.optional-dependencies]`, not core `dependencies`), so `import
pandas` must never appear anywhere in `feature_engine/`, not at module level
and not locally/lazily inside a function either — importing the module
itself would break a polars-only install regardless of which class is used.

Backend checks go through `narwhals.dependencies` (`nwd.is_pandas_dataframe`,
`nwd.is_pandas_series`, `nwd.is_pandas_index`, `nwd.is_into_series`, etc.).
Once a branch is confirmed pandas, call its methods/attributes directly on
the object already in hand (`.loc`, `.columns`, `.index`, `.select_dtypes`,
...) — no import needed for that, since Python only needs a module imported
to reference the module itself (`pd.something`), not to call methods on an
object that's already an instance of that module's class.

## Transformer code

- `check_X` and `check_X_y` return a narwhals dataframe. Bind it as
  `nw_X = check_X(X)`, pass the native `X` to the variable, NaN and feature-name
  helpers, compute on `nw_X`, and return `.to_native()`.
- When pandas keeps a native fast path, branch with
  `if nwd.is_pandas_dataframe(X):`, comment it with
  `# pandas is faster than narwhals.` and put the narwhals code in `else`.
- To use the target together with `X`, call `add_target_to_X(nw_X, y)` from
  `feature_engine/encoding/_helper_functions.py` and read it with `TARGET_NAME`.
  It works for series, list and array targets, and with pandas the column takes
  the index of `X`. Don't write this pairing again in a transformer.
- In the narwhals path, prefer narwhals expressions over Python loops on grouped
  results: aggregate with simple aggregations, then combine the columns in a
  `select`.
- narwhals expressions need string column names, while pandas allows integers.
  Take such columns with `get_column` and rename them before using expressions.
- Name temporary columns with double underscores (`__mean__`, `__count__`) so
  they can't clash with the user's columns.
- Don't add `# type: ignore`. If mypy complains because a parameter typed
  `Optional` is reassigned, store the value under a new name instead (for
  example `y_pd`).

## Init parameters

- Validate parameters in `__init__` only. Don't check them again in `fit` or
  `transform`, and don't test for errors raised by changing an attribute after
  init.
- For parameters that take a set of strings, check the type before the
  membership test, so lists, tuples, `None` and numbers raise the same error:

  ```python
  if not isinstance(encoding_method, str) or encoding_method not in [
      "ordered",
      "arbitrary",
  ]:
  ```

- Error messages follow the scikit-learn convention and end with
  `f"Got {param} instead."`.

## Booleans and control flow

- Compare booleans explicitly: `if x is True:` / `if x is False:`, never
  `if x:` / `if not x:`.
- Check container emptiness with `len(x) == 0`, never `if not x:`.
- `isinstance(...)` checks and `in`/`not in` membership tests are already
  explicit — leave them as-is, this rule isn't about those.
- The explicit `is True`/`is False` comparison is for flow control
  (`if`/`while` conditions) only — don't tack it onto a variable
  assignment.
- Call boolean checks such as `nwd.is_pandas_dataframe(X)` directly in the
  condition, `if nwd.is_pandas_dataframe(X) is True:`, instead of storing the
  result in a variable (`is_pandas = ...`) and testing that later.

## Comments

One line, two at most, in source code and tests. Only explain a non-obvious
WHY (a hidden constraint, a subtle backend difference, a workaround) — what the
reader needs to know about the code — never describe WHAT the code does.

## Don't anticipate errors

Don't add error handling or validation for scenarios that can't happen. If
unsure whether something can happen, check it (grep, run a quick repro) or
ask — don't guess and defensively code around it.

## Redundant lists/sets

- Narwhals' `.columns` is already `list[str]` — don't wrap it in `list()`.
- pandas' `.columns` is an `Index`, not a list — `list()` is required there
  (an `Index == list` comparison is elementwise, not a clean bool).

## Keep tests passing when you change a function or class

Whenever you change a function or class, run its corresponding tests. If
they fail, resolve it — don't leave it — by figuring out whether the test
needs updating (e.g. it exercised behavior that's no longer supported) or
the implementation has a real bug, and fixing whichever one is wrong.

## Keep docs in sync with transformer changes

When new functionality is introduced in a transformer, update its
corresponding `docs/user_guide/<module>/<ClassName>.rst` with a short
worked example showing the new functionality. When behaviour changes, check
that the outputs shown in the user guide examples are still correct.

User guides and other user-facing documentation are written for users:
assume readers don't know the source code, and certainly not narwhals. Explain
what a feature does and when to use it, in plain terms, without implementation
details or references to how the code used to behave.

## Verify before applying

Benchmark before claiming a speedup, and diff old-vs-new output across
realistic and edge cases (empty/all-NaN, both backends, both dtype
branches) before trusting a rewrite — logic mistakes here are easy to make
and easy to miss without an actual comparison. Compare like with like: time
the same work (for example the whole `fit()`) before and after.

Benchmark a range of data sizes, but base the decision mainly on the sizes
each backend is typically used with: 10k to 500k rows for pandas, and 500k
rows and more for polars. Smaller and larger sizes are worth measuring, but
they weigh less in the decision.

## Tests

Every transformer test file has the same structure, so they are easy to
maintain:

```python
# init parameters
def test_error_if_<param>_not_allowed(...)   # one test per error message
def test_init_param_assignment(...)          # several valid value combinations

# fit and transform
...
```

- Init error tests are parametrized with wrong values and wrong types.
- `test_init_param_assignment` checks every init parameter except `variables`
  and `return_empty`, which are tested elsewhere.
- Fit and transform tests don't assert init parameters.
- Every `pytest.raises` and `pytest.warns` matches the full message with
  `match=re.escape(msg)`, including `NotFittedError` and messages that come
  from scikit-learn. Never use
  `with pytest.raises() as record: ... assert str(record.value) == msg`.
  Matching the full message catches tests that pass for the wrong reason.

Backends and data:

- Dataframe-agnostic means one test, both backends: request the `make_df`
  fixture from `tests/conftest.py`, which runs the test with `pd.DataFrame`
  and `pl.DataFrame`, and assert the same input produces the same output
  values on both. Never write a separate pandas-only test and a
  separate polars-only test for the same behavior — that duplicates
  the test and hides the point of being dataframe-agnostic, which is
  that the same input gives the same output regardless of backend.
  Keep a test single-backend only when the behavior itself is
  backend-specific (e.g. integer column names, which polars doesn't
  support; pandas category or nullable extension dtypes), and check those
  with `pd.testing.assert_frame_equal`.
- Use the helpers in `tests/backend_helpers.py`: `frame_to_dict`, `null_count`
  and `make_series`. Don't add per-file helpers that do the same.
- Data used by several test files of a module lives in that module's
  `conftest.py`, as fixtures that return plain dicts, with `None` for missing
  values. Data used by one file stays in that file.
- Pass the target as a series built with `make_series`, and add one test with
  the target as a list and as a numpy array.
- Check outputs with `assert isinstance(Xt, make_df)` and compare
  `frame_to_dict(Xt)` with a dict. Compare floats with `pytest.approx`.
- Don't call polars' `to_pandas()` in tests: pyarrow is not installed locally
  or in CI.
- Name helpers after what they return (`frame_to_dict`, not `_cols`).

## API changes

- New parameters default to preserve current behavior.
- When adding a parameter to a function called from multiple sites (or a
  shared private helper), thread it through every call site, not just the
  one you're looking at.

## Before pushing

- Run the tests of the changed code, `flake8 feature_engine tests` (lines of 88
  characters at most) and `mypy feature_engine`. Running mypy on single files
  ignores the exclusions in `pyproject.toml`.
- If the target branch already has failing tests, compare the failing tests
  before and after the change instead of expecting a clean run.

## Pull requests

- When a PR is built on another open PR and that one is squash-merged, rebase
  with `git rebase --onto origin/<target branch> <old base tip>` so the PR shows
  only its own files. Push with `--force-with-lease`.
- Don't end PR descriptions with an AI tool attribution line, such as
  "Generated with Claude Code".
