"""Functions to compute the text features of string variables.

TextFeatures calls these functions instead of handling each dataframe library
itself.

All functions take the dataframe returned by `check_X`: a pandas dataframe, or a
narwhals dataframe for the other libraries.
"""

import string
from functools import cached_property
from typing import Any, List, Sequence, Union

import narwhals as nw
from narwhals.typing import IntoDataFrame

from feature_engine.dataframe_helpers import (
    add_category_to_df,
    add_columns_to_df,
    missing_values_count,
)

DataFrame = Union[IntoDataFrame, nw.DataFrame]

# The characters Python treats as whitespace. Listed explicitly because the \s of
# polars' regex engine misses \x1c-\x1f, and we want the same counts everywhere.
_WHITESPACE = (
    "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004"
    "\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)
_WORD = f"[^{_WHITESPACE}]+"

# Each feature is computed from the text statistics of one of the classes below,
# so all backends share the same definitions.
TEXT_FEATURES = {
    "char_count": lambda t: t.length - t.count_in(_WHITESPACE),
    "word_count": lambda t: t.word_count,
    "sentence_count": lambda t: t.count(r"[.!?]+"),
    "avg_word_length": lambda t: (t.length - t.count_in(_WHITESPACE))
    / t.word_count.clip(1),
    "digit_count": lambda t: t.count(r"\d"),
    "letter_count": lambda t: t.count_in(string.ascii_letters),
    "uppercase_count": lambda t: t.count(r"[A-Z]"),
    "lowercase_count": lambda t: t.count_in(string.ascii_lowercase),
    "special_char_count": lambda t: t.count_not_in(
        string.ascii_letters + string.digits + _WHITESPACE
    ),
    "whitespace_count": lambda t: t.count_in(_WHITESPACE),
    "whitespace_ratio": lambda t: t.count_in(_WHITESPACE) / t.length.clip(1),
    "digit_ratio": lambda t: t.count(r"\d")
    / (t.length - t.count_in(_WHITESPACE)).clip(1),
    "uppercase_ratio": lambda t: t.count(r"[A-Z]")
    / (t.length - t.count_in(_WHITESPACE)).clip(1),
    "has_digits": lambda t: t.contains(r"\d"),
    "has_uppercase": lambda t: t.contains(r"[A-Z]"),
    "is_empty": lambda t: t.is_empty(),
    "starts_with_uppercase": lambda t: t.contains(r"^[A-Z]"),
    "ends_with_punctuation": lambda t: t.ends_with_punctuation(),
    "unique_word_count": lambda t: t.unique_word_count,
    "lexical_diversity": lambda t: t.unique_word_count / t.word_count.clip(1),
}


class _PandasText:
    """Text statistics of a pandas Series of strings."""

    def __init__(self, text, native_namespace):
        self.text = text
        self._pd = native_namespace
        # several features share the same counts, and pandas computes them eagerly
        self._counts: dict = {}

    @cached_property
    def length(self):
        return self.text.str.len()

    @cached_property
    def word_count(self):
        # a Python loop is 2x faster than pandas' str.split().str.len()
        words = [len(s.split()) for s in self.text.tolist()]
        return self._pd.Series(words, index=self.text.index)

    @cached_property
    def unique_word_count(self):
        words = [len(set(s.lower().split())) for s in self.text.tolist()]
        return self._pd.Series(words, index=self.text.index)

    def count(self, pattern):
        if ("count", pattern) not in self._counts:
            self._counts[("count", pattern)] = self.text.str.count(pattern)
        return self._counts[("count", pattern)]

    def count_in(self, characters):
        # deleting the characters with translate is faster than a regex count
        if ("count_in", characters) not in self._counts:
            self._counts[("count_in", characters)] = (
                self.length - self.count_not_in(characters)
            )
        return self._counts[("count_in", characters)]

    def count_not_in(self, characters):
        table = str.maketrans("", "", characters)
        return self.text.str.translate(table).str.len()

    def contains(self, pattern):
        return self.text.str.contains(pattern, regex=True).astype(int)

    def is_empty(self):
        return self.text.eq("").astype(int)

    def ends_with_punctuation(self):
        return self.text.str.match(r".*[.!?]$").astype(int)


class _NarwhalsText:
    """Text statistics of a string column, as narwhals expressions."""

    def __init__(self, text, namespace):
        self.text = text
        self._ns = namespace

    @property
    def length(self):
        return self.text.str.len_chars().cast(self._ns.Int64)

    @property
    def word_count(self):
        return self.count(_WORD)

    @property
    def unique_word_count(self):
        words = (
            self.text.str.to_lowercase()
            .str.replace_all(f"[{_WHITESPACE}]+", " ")
            .str.strip_chars(" ")
            .str.split(" ")
        )
        # splitting a text without words returns one empty word
        return (
            self._ns.when(self.word_count == 0)
            .then(0)
            .otherwise(words.list.unique().list.len())
            .cast(self._ns.Int64)
        )

    def count(self, pattern):
        # narwhals can't count matches, but replacing each match with 2
        # characters instead of 1 makes the text 1 character longer per match
        return (
            self.text.str.replace_all(pattern, "ab").str.len_chars()
            - self.text.str.replace_all(pattern, "a").str.len_chars()
        ).cast(self._ns.Int64)

    def count_in(self, characters):
        return self.length - self.count_not_in(characters)

    def count_not_in(self, characters):
        kept = self.text.str.replace_all(f"[{characters}]", "")
        return kept.str.len_chars().cast(self._ns.Int64)

    def contains(self, pattern):
        return self.text.str.contains(pattern).cast(self._ns.Int64)

    def is_empty(self):
        return (self.text.str.len_chars() == 0).cast(self._ns.Int64)

    def ends_with_punctuation(self):
        # Python's regex $ also matches before a final \n, which would let
        # "x.\n\n" match on backends that use it
        ends = self.text.str.contains(r"^[^\n]*[.!?]\n?$")
        return (ends & ~self.text.str.ends_with("\n\n")).cast(self._ns.Int64)


class _PolarsText(_NarwhalsText):
    """Text statistics of a string column, as polars expressions."""

    @property
    def unique_word_count(self):
        words = self.text.str.to_lowercase().str.extract_all(_WORD)
        return words.list.n_unique().cast(self._ns.Int64)

    def count(self, pattern):
        return self.text.str.count_matches(pattern).cast(self._ns.Int64)

    def count_not_in(self, characters):
        return self.count(f"[^{characters}]")


def find_non_text_variables(X: DataFrame, variables: Sequence[str]) -> List[str]:
    """
    Returns the variables that are not strings, objects or categories of strings.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables to check.

    Returns
    -------
    variables : list
        The names of the variables that don't contain text.
    """
    nw_X = nw.from_native(X, eager_only=True)
    non_text = []
    for var in variables:
        dtype = nw_X.get_column(var).dtype
        # pandas categories can be numbers, polars categories are always strings
        if isinstance(dtype, nw.Categorical) and not isinstance(X, nw.DataFrame):
            is_text = X[var].cat.categories.inferred_type == "string"
        else:
            is_text = isinstance(
                dtype, (nw.String, nw.Object, nw.Categorical, nw.Enum)
            )
        if is_text is False:
            non_text.append(var)
    return non_text


def add_text_features(
    X: DataFrame,
    variables: Sequence[str],
    features: Sequence[str],
    fill_missing: bool,
) -> DataFrame:
    """
    Adds the text features of each variable to the end of the dataframe.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the text variables.

    features : list
        The names of the features to compute, keys of `TEXT_FEATURES`.

    fill_missing : bool
        Whether to replace the missing values of the variables with empty strings.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with one new column per variable and feature, named
        `<variable>_<feature>`.
    """
    # pandas is faster than narwhals, which can't split strings without pyarrow.
    if not isinstance(X, nw.DataFrame):
        if fill_missing is True:
            counts = missing_values_count(X, variables)
            with_na = [var for var, n in zip(variables, counts) if n > 0]
            # pandas categorical columns only take values that are categories
            X = add_category_to_df(X, with_na, "")
            X = X.fillna({var: "" for var in with_na})
        new_features = {}
        for var in variables:
            statistics = _PandasText(X[var], nw.get_native_namespace(X))
            for feature in features:
                values = TEXT_FEATURES[feature](statistics)
                new_features[f"{var}_{feature}"] = values.to_numpy()
        return add_columns_to_df(X, new_features)
    elif X.implementation is nw.Implementation.POLARS:
        # polars counts regex matches natively, narwhals needs two replacements.
        X_pl = _add_features_with_expressions(
            X.to_native(),
            variables,
            features,
            fill_missing,
            nw.get_native_namespace(X),
            _PolarsText,
        )
        return nw.from_native(X_pl, eager_only=True)
    else:
        return _add_features_with_expressions(
            X, variables, features, fill_missing, nw, _NarwhalsText
        )


def _add_features_with_expressions(
    X: Any,
    variables: Sequence[str],
    features: Sequence[str],
    fill_missing: bool,
    namespace: Any,
    text_class: Any,
) -> Any:
    # polars and narwhals expressions share the API used here
    filled_text, new_features = [], []
    for var in variables:
        if fill_missing is True:
            filled_text.append(namespace.col(var).fill_null(""))
        text = namespace.col(var).cast(namespace.String).fill_null("")
        statistics = text_class(text, namespace)
        new_features += [
            TEXT_FEATURES[feature](statistics).alias(f"{var}_{feature}")
            for feature in features
        ]
    return X.with_columns(*filled_text, *new_features)
