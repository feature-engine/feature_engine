# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import copy
import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.encoding._docstrings import (
    _errors_docstring,
    _ignore_format_docstring,
    _transform_docstring,
    _variables_docstring,
)
from feature_engine.encoding.base_encoder import (
    CategoricalInitExpandedMixin,
    CategoricalMethodsMixin,
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_docstring,
    errors=_errors_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    transform=_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class CountFrequencyEncoder(CategoricalInitExpandedMixin, CategoricalMethodsMixin):
    """
    The CountFrequencyEncoder() replaces categories by either the count or the
    percentage of observations per category.

    For example in the variable colour, if 10 observations are blue, blue will
    be replaced by 10. Alternatively, if 10% of the observations are blue, blue
    will be replaced by 0.1.

    The CountFrequencyEncoder() will encode only categorical variables by default
    (type 'object' or 'categorical'). You can pass a list of variables to encode.
    Alternatively, the encoder will find and encode all categorical variables
    (type 'object' or 'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the counts or frequencies for each
    variable (fit). The encoder then replaces the categories with those numbers
    (transform).

    More details in the :ref:`User Guide <count_freq_encoder>`.

    Parameters
    ----------
    encoding_method: str, default='count'
        Desired method of encoding.

        **'count'**: number of observations per category

        **'frequency'**: percentage of observations per category

    {variables}

    {ignore_format}

    {errors}

    threshold: int,float or None, default=None
        if a category is present in less than threshold number / fraction
        of observations, it will be grouped with other labels, and then replaced
        by the number/fraction of observations seen in all this rare labels combined.
        If encoding_method is 'count' then threshold takes an integer.
        If encoding_method is frequency, then threshold takes a float.
        If None, no grouping will be performed

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the count or frequency per category, per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the count or frequency per category, per variable.

    {fit_transform}

    {inverse_transform}

    {transform}

    Notes
    -----
    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    There is a similar implementation in the the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    category_encoders.count.CountEncoder
    """

    def __init__(
        self,
        encoding_method: str = "count",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
        errors: str = "ignore",
        threshold: Union[int, float, None] = None,
    ) -> None:

        if encoding_method not in ["count", "frequency"]:
            raise ValueError(
                "encoding_method takes only values 'count' and 'frequency'"
            )

        if (
            encoding_method == "count"
            and not (isinstance(threshold, int))
            and threshold is not None
        ):
            raise ValueError(
                "threshold expects int for encoding=count but got {0}".format(
                    type(threshold)
                )
            )

        if (
            encoding_method == "frequency"
            and not (isinstance(threshold, float))
            and threshold is not None
        ):
            raise ValueError(
                "threshold expects float for encoding=frequency but got {0}".format(
                    type(threshold)
                )
            )

        super().__init__(variables, ignore_format, errors)
        self.encoding_method = encoding_method
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the counts or frequencies which will be used to replace the categories.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: pandas Series, default = None
            y is not needed in this encoder. You can pass y or None.

        """

        X = check_X(X)
        self._check_or_select_variables(X)
        self._get_feature_names_in(X)

        self.encoder_dict_ = {}

        # Remember unique values for each feature
        # for validation inside transform
        self.__hold_unique_feature_values = {}

        if self.encoding_method == "count":
            for var in self.variables_:
                self.encoder_dict_[var] = X[var].value_counts().to_dict()
                self.__hold_unique_feature_values[var] = X[var].unique().tolist()

        elif self.encoding_method == "frequency":
            for var in self.variables_:
                n_obs = float(len(X))
                self.encoder_dict_[var] = (X[var].value_counts() / n_obs).to_dict()
                self.__hold_unique_feature_values[var] = X[var].unique().tolist()

        if self.threshold is not None:
            enc_dct_cpy = copy.deepcopy(self.encoder_dict_.copy())
            for k, v in enc_dct_cpy.items():
                self.encoder_dict_[k].update(
                    {
                        "backoff_bin": sum(
                            [
                                self.encoder_dict_[k].pop(key)
                                for key in [
                                    key
                                    for key, value in v.items()
                                    if value <= self.threshold
                                ]
                            ]
                        )
                    }
                )

        self._check_encoding_dictionary()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace categories with the learned parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The dataset to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing the categories replaced by numbers.
        """

        X = self._check_transform_input_and_state(X)

        for feature in self.variables_:
            keys_from_encoder_dict = set(self.encoder_dict_[feature].keys())
            keys_from_fit = set(self.__hold_unique_feature_values[feature])
            reminder_keys = keys_from_fit - keys_from_encoder_dict
            X[feature] = X[feature].replace(reminder_keys, "backoff_bin")

        return super().transform(X)

    def inverse_transform(self, X: pd.DataFrame):
        """inverse_transform is not implemented for this transformer."""
        return self
