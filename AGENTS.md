# AGENTS.md

Conventions for working in this repo. Optimize for readability and speed, 
in that order of how you decide, but don't ship a slow default when a 
fast one is free.

## Inputs

Feature-engine transformers take dataframes (pandas, polars, or any other
narwhals-supported backend) as input, not numpy arrays. Don't add
handling for array input.

## Booleans and control flow

- Compare booleans explicitly: `if x is True:` / `if x is False:`, never
  `if x:` / `if not x:`.
- Check container emptiness with `len(x) == 0`, never `if not x:`.
- `isinstance(...)` checks and `in`/`not in` membership tests are already
  explicit — leave them as-is, this rule isn't about those.

## Comments

Max 2 lines. Only explain a non-obvious WHY (a hidden constraint, a subtle
backend difference, a workaround) — never describe WHAT the code does.

## Don't anticipate errors

Don't add error handling or validation for scenarios that can't happen. If
unsure whether something can happen, check it (grep, run a quick repro) or
ask — don't guess and defensively code around it.

## Redundant lists/sets

- Narwhals' `.columns` is already `list[str]` — don't wrap it in `list()`.
- pandas' `.columns` is an `Index`, not a list — `list()` is required there
  (an `Index == list` comparison is elementwise, not a clean bool).

## Verify before applying

Benchmark before claiming a speedup, and diff old-vs-new output across
realistic and edge cases (empty/all-NaN, both backends, both dtype
branches) before trusting a rewrite — logic mistakes here are easy to make
and easy to miss without an actual comparison.

## Tests

- `pytest.raises(ExceptionType, match=msg)`, never
  `with pytest.raises() as record: ... assert str(record.value) == msg`.

## API changes

- New parameters default to preserve current behavior.
- When adding a parameter to a function called from multiple sites (or a
  shared private helper), thread it through every call site, not just the
  one you're looking at.
