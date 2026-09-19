# Authors: Ankit Hemant Lade (contributor)
# License: BSD 3 clause
import string
from functools import cached_property
from typing import List, Optional, Union, cast

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_drop_original,
    _check_param_missing_values,
)
from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)

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
    "avg_word_length": lambda t: t.strip_length() / t.word_count.clip(1),
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

    def strip_length(self):
        return self.text.str.strip().str.len()

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

    def strip_length(self):
        return self.text.str.strip_chars(_WHITESPACE).str.len_chars()

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


class TextFeatures(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    TextFeatures() extracts numerical features from text/string variables. This
    transformer is useful for extracting basic text statistics that can be used
    as features in machine learning models.

    A list with the text variables must be passed as an argument.

    More details in the :ref:`User Guide <text_features>`.

    Parameters
    ----------
    variables: string, list
        The list of text/string variables to extract features from.

    features: list, default=None
        List of text features to extract. Available features are:

        - 'char_count': Number of characters, excluding whitespace
        - 'word_count': Number of words (whitespace-separated tokens)
        - 'sentence_count': Number of sentences (based on .!? punctuation)
        - 'avg_word_length': Number of characters, from the first to the last
          non-whitespace character, divided by the number of words
        - 'digit_count': Number of digit characters
        - 'letter_count': Number of letters a-z and A-Z
        - 'uppercase_count': Number of uppercase letters A-Z
        - 'lowercase_count': Number of lowercase letters a-z
        - 'special_char_count': Number of characters that are not a-z, A-Z, 0-9
          or whitespace
        - 'whitespace_count': Number of whitespace characters
        - 'whitespace_ratio': Ratio of whitespace to total characters
        - 'digit_ratio': Ratio of digits to non-whitespace characters
        - 'uppercase_ratio': Ratio of uppercase letters to non-whitespace
          characters
        - 'has_digits': Binary indicator if text contains digits
        - 'has_uppercase': Binary indicator if text contains uppercase letters A-Z
        - 'is_empty': Binary indicator if text is empty
        - 'starts_with_uppercase': Binary indicator if text starts with A-Z
        - 'ends_with_punctuation': Binary indicator if text ends with .!?
        - 'unique_word_count': Number of unique words (case-insensitive)
        - 'lexical_diversity': Ratio of unique words to total words

        If None, extracts all available features.

    missing_values: string, default='ignore'
        If 'ignore', missing values will be filled with an empty string before
        feature extraction. If 'raise', the transformer will raise an error if
        missing data is found.

    drop_original: bool, default=False
        Whether to drop the original text columns after transformation.

    Attributes
    ----------
    variables_:
        The list of text variables that will be transformed.

    features_:
        The list of features that will be extracted.

    feature_names_in_:
        List with the names of features seen during fit.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        This transformer does not learn parameters.

    fit_transform:
        Fit to data, then transform it.

    transform:
        Extract text features and add them to the dataframe.

    get_feature_names_out:
        Get output feature names for transformation.

    See Also
    --------
    feature_engine.encoding.StringSimilarityEncoder :
        Encodes categorical variables based on string similarity.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.text import TextFeatures
    >>> X = pd.DataFrame({
    ...     'text': ['Hello World!', 'Python is GREAT.', 'ML rocks 123']
    ... })
    >>> tf = TextFeatures(
    ...     variables=['text'],
    ...     features=['char_count', 'word_count', 'has_digits']
    ... )
    >>> tf.fit(X)
    TextFeatures(features=['char_count', 'word_count', 'has_digits'],
                 variables=['text'])
    >>> X = tf.transform(X)
    >>> pd.options.display.max_columns = 10
    >>> print(X)
                   text  text_char_count  text_word_count  text_has_digits
    0      Hello World!               11                2                0
    1  Python is GREAT.               14                3                0
    2      ML rocks 123               10                3                1
    """

    def __init__(
        self,
        variables: Union[str, List[str]],
        features: Optional[List[str]] = None,
        missing_values: str = "ignore",
        drop_original: bool = False,
    ) -> None:

        if isinstance(variables, str):
            variables = [variables]
        if not isinstance(variables, list) or not all(
            isinstance(v, str) for v in variables
        ):
            raise ValueError(
                "variables must be a string or a list of strings. "
                f"Got {variables} instead."
            )

        if features is not None and (
            not isinstance(features, list)
            or not all(isinstance(f, str) and f in TEXT_FEATURES for f in features)
        ):
            raise ValueError(
                "features must be None or a list with any of "
                f"{list(TEXT_FEATURES.keys())}. Got {features} instead."
            )

        _check_param_drop_original(drop_original)
        _check_param_missing_values(missing_values)

        self.variables = variables
        self.features = features
        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: IntoDataFrame, y=None):
        """
        This transformer does not learn any parameters.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: Series, or np.array. Defaults to None.
            The target. It is not needed in this transformer. You can pass y or None.
        """
        nw_X = check_X(X)

        missing = set(self.variables) - set(nw_X.columns)
        if len(missing) > 0:
            raise ValueError(f"Variables {missing} are not present in the dataframe.")

        non_text = []
        for var in self.variables:
            dtype = nw_X.get_column(var).dtype
            # pandas categories can be numbers, polars categories are always strings
            if isinstance(dtype, nw.Categorical) and nwd.is_pandas_dataframe(X) is True:
                is_text = X[var].cat.categories.inferred_type == "string"
            else:
                is_text = isinstance(
                    dtype, (nw.String, nw.Object, nw.Categorical, nw.Enum)
                )
            if is_text is False:
                non_text.append(var)
        if len(non_text) > 0:
            raise ValueError(
                f"Variables {non_text} are not object or string. "
                "Please provide text variables only."
            )

        self.variables_ = self.variables

        if self.missing_values == "raise":
            _check_contains_na(
                X, cast(list[Union[str, int]], self.variables_), error_msg="optional"
            )

        if self.features is None:
            self.features_ = list(TEXT_FEATURES.keys())
        else:
            self.features_ = self.features

        self.feature_names_in_ = nw_X.columns
        self.n_features_in_ = nw_X.shape[1]

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Extract text features and add them to the dataframe.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe
            The dataframe with the original columns plus the new text features.
        """
        check_is_fitted(self)
        nw_X = check_X(X)
        _check_X_matches_training_df(nw_X, self.n_features_in_)

        if self.missing_values == "raise":
            _check_contains_na(
                X, cast(list[Union[str, int]], self.variables_), error_msg="optional"
            )

        # pandas is faster than narwhals.
        if nwd.is_pandas_dataframe(X) is True:
            X_new = self._transform_pandas(X, nw.get_native_namespace(nw_X))
        elif nwd.is_polars_dataframe(X) is True:
            # polars counts regex matches natively, narwhals needs two replacements.
            X_new = self._transform_expressions(
                X, nw.get_native_namespace(nw_X), _PolarsText
            )
        else:
            X_new = self._transform_expressions(nw_X, nw, _NarwhalsText).to_native()

        return X_new

    def _transform_pandas(self, X, native_namespace):
        X_new = X[self.feature_names_in_]
        if self.missing_values == "ignore":
            X_new = X_new.fillna({var: "" for var in self.variables_})

        new_features = []
        for var in self.variables_:
            statistics = _PandasText(X_new[var], native_namespace)
            new_features += [
                TEXT_FEATURES[feature](statistics).rename(f"{var}_{feature}")
                for feature in self.features_
            ]

        X_new = native_namespace.concat([X_new, *new_features], axis=1)
        if self.drop_original is True:
            X_new = X_new.drop(columns=self.variables_)

        return X_new

    def _transform_expressions(self, X, namespace, text_class):
        # polars and narwhals expressions share the API used here
        filled_text, new_features = [], []
        for var in self.variables_:
            if self.missing_values == "ignore":
                filled_text.append(namespace.col(var).fill_null(""))
            text = namespace.col(var).cast(namespace.String).fill_null("")
            statistics = text_class(text, namespace)
            new_features += [
                TEXT_FEATURES[feature](statistics).alias(f"{var}_{feature}")
                for feature in self.features_
            ]

        X_new = X.select(self.feature_names_in_).with_columns(
            *filled_text, *new_features
        )
        if self.drop_original is True:
            X_new = X_new.drop(self.variables_)

        return X_new

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. If ``None``, uses ``feature_names_in_``.

        Returns
        -------
        feature_names_out : list of str
            Output feature names.
        """
        check_is_fitted(self)

        # Start with original features
        if self.drop_original is True:
            feature_names = [
                f for f in self.feature_names_in_ if f not in self.variables_
            ]
        else:
            feature_names = list(self.feature_names_in_)

        # Add new text feature names
        for var in self.variables_:
            for feature_name in self.features_:
                feature_names.append(f"{var}_{feature_name}")

        return feature_names
