# Authors: Ankit Hemant Lade (contributor)
# License: BSD 3 clause
from typing import List, Optional, Union, cast

import pandas as pd
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
from feature_engine.tags import _return_tags

# Available text features and their computation functions
TEXT_FEATURES = {
    "char_count": lambda x: x.str.replace(r"\s+", "", regex=True).str.len(),
    "word_count": lambda x: x.str.strip().str.split().str.len(),
    "sentence_count": lambda x: x.str.count(r"[.!?]+"),
    "avg_word_length": lambda x: (
        x.str.strip().str.len() / x.str.strip().str.split().str.len()
    ).fillna(0),
    "digit_count": lambda x: x.str.count(r"\d"),
    "letter_count": lambda x: x.str.count(r"[a-zA-Z]"),
    "uppercase_count": lambda x: x.str.count(r"[A-Z]"),
    "lowercase_count": lambda x: x.str.count(r"[a-z]"),
    "special_char_count": lambda x: x.str.count(r"[^a-zA-Z0-9\s]"),
    "whitespace_count": lambda x: x.str.count(r"\s"),
    "whitespace_ratio": lambda x: x.str.count(r"\s") / x.str.len().replace(0, 1),
    "digit_ratio": lambda x: x.str.count(r"\d")
    / x.str.replace(r"\s+", "", regex=True).str.len().replace(0, 1),
    "uppercase_ratio": lambda x: x.str.count(r"[A-Z]")
    / x.str.replace(r"\s+", "", regex=True).str.len().replace(0, 1),
    "has_digits": lambda x: x.str.contains(r"\d", regex=True).astype(int),
    "has_uppercase": lambda x: x.str.contains(r"[A-Z]", regex=True).astype(int),
    "is_empty": lambda x: (x.str.len() == 0).astype(int),
    "starts_with_uppercase": lambda x: x.str.match(r"^[A-Z]").astype(int),
    "ends_with_punctuation": lambda x: x.str.match(r".*[.!?]$").astype(int),
    "unique_word_count": lambda x: (x.str.lower().str.split().apply(set).str.len()),
    "lexical_diversity": lambda x: (
        x.str.strip().str.split().str.len()
        / x.str.lower().str.split().apply(set).str.len()
    ).fillna(0),
}


class TextFeatures(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    TextFeatures() extracts numerical features from text/string variables. This
    transformer is useful for extracting basic text statistics that can be used
    as features in machine learning models.

    A list of variables must be passed as an argument.

    More details in the :ref:`User Guide <text_features>`.

    Parameters
    ----------
    variables: string, list
        The list of text/string variables to extract features from.

    features: list, default=None
        List of text features to extract. Available features are:

        - 'char_count': Number of characters in the text
        - 'word_count': Number of words (whitespace-separated tokens)
        - 'sentence_count': Number of sentences (based on .!? punctuation)
        - 'avg_word_length': Average length of words
        - 'digit_count': Number of digit characters
        - 'letter_count': Number of alphabetic characters (a-z, A-Z)
        - 'uppercase_count': Number of uppercase letters
        - 'lowercase_count': Number of lowercase letters
        - 'special_char_count': Number of special characters (non-alphanumeric)
        - 'whitespace_count': Number of whitespace characters
        - 'whitespace_ratio': Ratio of whitespace to total characters
        - 'digit_ratio': Ratio of digits to total characters
        - 'uppercase_ratio': Ratio of uppercase to total characters
        - 'has_digits': Binary indicator if text contains digits
        - 'has_uppercase': Binary indicator if text contains uppercase
        - 'is_empty': Binary indicator if text is empty
        - 'starts_with_uppercase': Binary indicator if text starts with uppercase
        - 'ends_with_punctuation': Binary indicator if text ends with .!?
        - 'unique_word_count': Number of unique words (case-insensitive)
        - 'lexical_diversity': Ratio of unique words to total words

        If None, extracts all available features.

    missing_values: string, default='ignore'
        If 'ignore', NaNs will be filled with an empty string before feature
        extraction. If 'raise', the transformer will raise an error if missing data
        is found.

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

        # Validate variables
        if isinstance(variables, str):
            variables = [variables]
        if not isinstance(variables, list) or not all(
            isinstance(v, str) for v in variables
        ):
            raise ValueError(
                "variables must be a string or a list of strings. "
                f"Got {type(variables).__name__} instead."
            )

        # Validate features
        if features is not None:
            if not isinstance(features, list) or not all(
                isinstance(f, str) for f in features
            ):
                raise ValueError(
                    "features must be None or a list of strings. "
                    f"Got {type(features).__name__} instead."
                )
            invalid_features = set(features) - set(TEXT_FEATURES.keys())
            if invalid_features:
                raise ValueError(
                    f"Invalid features: {invalid_features}. "
                    f"Available features are: {list(TEXT_FEATURES.keys())}"
                )

        _check_param_drop_original(drop_original)
        _check_param_missing_values(missing_values)

        self.variables = variables
        self.features = features
        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameters.

        Stores feature names and validates that the specified variables are
        present.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, or np.array. Defaults to None.
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = check_X(X)

        # Validate user-specified variables exist
        missing = set(self.variables) - set(X.columns)
        if missing:
            raise ValueError(f"Variables {missing} are not present in the dataframe.")

        # Validate that the variables are object or string
        non_text = [
            col
            for col in self.variables
            if not (
                pd.api.types.is_string_dtype(X[col])
                or pd.api.types.is_object_dtype(X[col])
            )
        ]
        if non_text:
            raise ValueError(
                f"Variables {non_text} are not object or string. "
                "Please provide text variables only."
            )

        self.variables_ = self.variables

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, cast(list[Union[str, int]], self.variables_))

        # Set features to extract
        if self.features is None:
            self.features_ = list(TEXT_FEATURES.keys())
        else:
            self.features_ = self.features

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract text features and add them to the dataframe.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe
            The dataframe with the original columns plus the new text features.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, cast(list[Union[str, int]], self.variables_))

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        # Fill NaN with empty string for feature extraction
        # This is safe because if missing_values is 'raise', it would have
        # raised an error above. So any remaining NaNs are either intended to
        # be filled or there are none.
        X[self.variables_] = X[self.variables_].fillna("")

        # Extract features for each text variable
        for var in self.variables_:
            for feature_name in self.features_:
                new_col_name = f"{var}_{feature_name}"
                feature_func = TEXT_FEATURES[feature_name]
                X[new_col_name] = feature_func(X[var])

                # Fill any NaN values resulting from computation with 0
                X[new_col_name] = X[new_col_name].fillna(0)

        if self.drop_original:
            X = X.drop(columns=self.variables_)

        return X

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
        if self.drop_original:
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
