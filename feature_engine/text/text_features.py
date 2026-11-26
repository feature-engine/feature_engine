# Authors: Ankit Hemant Lade (contributor)
# License: BSD 3 clause
from typing import List, Optional, Union, cast

import narwhals as nw
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
from feature_engine.dataframe_helpers import drop_variables_from_df, subset_df
from feature_engine.text._text_helpers import (
    TEXT_FEATURES,
    add_text_features,
    find_non_text_variables,
)


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
        - 'avg_word_length': Average number of characters per word
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
        X = check_X(X)

        missing = set(self.variables) - set(X.columns)
        if len(missing) > 0:
            raise ValueError(f"Variables {missing} are not present in the dataframe.")

        non_text = find_non_text_variables(X, self.variables)
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

        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

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
        X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)

        if self.missing_values == "raise":
            _check_contains_na(
                X, cast(list[Union[str, int]], self.variables_), error_msg="optional"
            )

        X = subset_df(X, self.feature_names_in_)
        X = add_text_features(
            X,
            self.variables_,
            self.features_,
            fill_missing=self.missing_values == "ignore",
        )
        if self.drop_original is True:
            X = drop_variables_from_df(X, self.variables_)

        return nw.to_native(X, pass_through=True)

    def _get_new_features_name(self) -> List[str]:
        """Return the names of the created features."""
        return [
            f"{var}_{feature}" for var in self.variables_ for feature in self.features_
        ]
