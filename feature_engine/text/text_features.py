# Authors: Ankit Hemant Lade (contributor)
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_drop_original,
)
from feature_engine.dataframe_checks import _check_X_matches_training_df, check_X
from feature_engine.tags import _return_tags

# Available text features and their computation functions
TEXT_FEATURES = {
    "char_count": lambda x: x.str.len(),
    "word_count": lambda x: x.str.split().str.len(),
    "sentence_count": lambda x: x.str.count(r"[.!?]+"),
    "avg_word_length": lambda x: x.apply(
        lambda s: sum(len(w) for w in str(s).split()) / max(len(str(s).split()), 1)
    ),
    "digit_count": lambda x: x.str.count(r"\d"),
    "uppercase_count": lambda x: x.str.count(r"[A-Z]"),
    "lowercase_count": lambda x: x.str.count(r"[a-z]"),
    "special_char_count": lambda x: x.str.count(r"[^a-zA-Z0-9\s]"),
    "whitespace_count": lambda x: x.str.count(r"\s"),
    "whitespace_ratio": lambda x: x.str.count(r"\s") / x.str.len().replace(0, 1),
    "digit_ratio": lambda x: x.str.count(r"\d") / x.str.len().replace(0, 1),
    "uppercase_ratio": lambda x: x.str.count(r"[A-Z]") / x.str.len().replace(0, 1),
    "has_digits": lambda x: x.str.contains(r"\d", regex=True).astype(int),
    "has_uppercase": lambda x: x.str.contains(r"[A-Z]", regex=True).astype(int),
    "is_empty": lambda x: (x.str.len() == 0).astype(int),
    "starts_with_uppercase": lambda x: x.str.match(r"^[A-Z]").astype(int),
    "ends_with_punctuation": lambda x: x.str.match(r".*[.!?]$").astype(int),
    "unique_word_count": lambda x: x.apply(lambda s: len(set(str(s).lower().split()))),
    "unique_word_ratio": lambda x: x.apply(
        lambda s: len(set(str(s).lower().split())) / max(len(str(s).split()), 1)
    ),
}


class TextFeatures(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    TextFeatures() extracts numerical features from text/string variables. This
    transformer is useful for extracting basic text statistics that can be used
    as features in machine learning models.

    The transformer can extract various text features including character counts,
    word counts, sentence counts, and various ratios and indicators.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type object (string).

    More details in the :ref:`User Guide <text_features>`.

    Parameters
    ----------
    variables: list, default=None
        The list of text/string variables to extract features from. If None, the
        transformer will automatically select all object (string) columns.

    features: list, default=None
        List of text features to extract. Available features are:

        - 'char_count': Number of characters in the text
        - 'word_count': Number of words (whitespace-separated tokens)
        - 'sentence_count': Number of sentences (based on .!? punctuation)
        - 'avg_word_length': Average length of words
        - 'digit_count': Number of digit characters
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
        - 'unique_word_ratio': Ratio of unique words to total words

        If None, extracts all available features.

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
        This transformer does not learn parameters. It stores the feature names
        and validates input.

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
    >>> tf = TextFeatures(features=['char_count', 'word_count', 'has_digits'])
    >>> tf.fit(X)
    >>> X = tf.transform(X)
    >>> X
                   text  text_char_count  text_word_count  text_has_digits
    0      Hello World!               12                2                0
    1  Python is GREAT.               16                3                0
    2       ML rocks 123               12                3                1
    """

    def __init__(
        self,
        variables: Union[None, str, List[str]] = None,
        features: Union[None, List[str]] = None,
        drop_original: bool = False,
    ) -> None:

        # Validate variables
        if variables is not None:
            if isinstance(variables, str):
                variables = [variables]
            elif not isinstance(variables, list) or not all(
                isinstance(v, str) for v in variables
            ):
                raise ValueError(
                    "variables must be None, a string, or a list of strings. "
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

        self.variables = variables
        self.features = features
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Stores feature names and validates that the specified variables are
        present and are of string/object type.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas Series, or np.array. Defaults to None.
            It is not needed in this transformer. You can pass y or None.

        Returns
        -------
        self: TextFeatures
            The fitted transformer.
        """

        # check input dataframe
        X = check_X(X)

        # Find or validate text variables
        if self.variables is None:
            # Select object/string columns
            self.variables_ = [col for col in X.columns if X[col].dtype == "object"]
            if len(self.variables_) == 0:
                raise ValueError(
                    "No object/string columns found in the dataframe. "
                    "Please specify variables explicitly."
                )
        else:
            # Validate user-specified variables exist
            missing = set(self.variables) - set(X.columns)
            if missing:
                raise ValueError(
                    f"Variables {missing} are not present in the dataframe."
                )
            self.variables_ = self.variables

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

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        # Extract features for each text variable
        for var in self.variables_:
            # Fill NaN with empty string for feature extraction
            text_col = X[var].fillna("")

            for feature_name in self.features_:
                new_col_name = f"{var}_{feature_name}"
                feature_func = TEXT_FEATURES[feature_name]
                X[new_col_name] = feature_func(text_col)

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
            Input features. If None, uses feature_names_in_.

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

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "categorical"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
