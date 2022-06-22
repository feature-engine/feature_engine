from typing import List, Optional, Union

import pandas as pd

from .._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from .._docstrings.substitute import Substitution
from ..encoding._docstrings import (
    _errors_docstring,
    _ignore_format_docstring,
    _variables_docstring,
)
from ..encoding.base_encoder import BaseCategorical


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_docstring,
    errors=_errors_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
)
class CategoryEncoder(BaseCategorical):
    """
    CategoryEncoder() ensures that categorical variables are encoded as pandas'
    'categorical' dtype instead of generic python 'object' or other dtypes.
    Under the hood, 'categorical' dtype is a representation that maps each
    category to an integer, thus providing a more memory-efficient object
    structure than e.g. 'str' and allowing faster grouping, mapping, and similar
    operations on the resulting object.

    This transformer remembers the encodings or levels that represent each
    category, and can thus be used to ensure that the correct encoding gets
    applied when passing categorical data to modeling packages that support this
    dtype, or to prevent unseen categories from reaching a further transformer
    or estimator in some pipeline, for example.

    Parameters
    ----------
    {variables}

    {ignore_format}

    {errors}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the category encodings assigned to each variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the ecodings or levels to use for each variable.

    transform:
        Encode the variables as categorical dtype
    """
    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
        errors: str = "ignore",
    ) -> None:
        super().__init__(variables, ignore_format, errors, errors != "raise")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the encodings or levels to use for representing categorical variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: pandas Series, default = None
            y is not needed in this encoder. You can pass y or None.
        """
        X = self._check_X(X)
        self._check_or_select_variables(X)
        self._get_feature_names_in(X)

        self.encoder_dict_ = dict()
        for var in self.variables_:
            self.encoder_dict_[var] = pd.Categorical(X[var]).categories

        self._check_encoding_dictionary()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables as pandas categorical dtype.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The dataset to encode.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing variables encoded as pandas categorical dtype.
        """
        X = self._check_transform_input_and_state(X)

        for feature, levels in self.encoder_dict_.items():
            X = X.assign(**{feature: pd.Categorical(X[feature], levels)})

        self._check_nas_in_result(X)
        return X
