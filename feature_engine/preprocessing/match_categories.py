import warnings
from typing import List, Optional, Union

import pandas as pd

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _missing_values_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.init_parameters.encoders import _ignore_format_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import _check_optional_contains_na, check_X
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixinNA,
    CategoricalMethodsMixin,
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    missing_values=_missing_values_docstring,
    variables=_variables_categorical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
)
class MatchCategories(
    CategoricalMethodsMixin, CategoricalInitMixinNA, GetFeatureNamesOutMixin
):
    """
    MatchCategories() ensures that categorical variables are encoded as pandas
    `'categorical'` dtype, instead of generic python `'object'` or other dtypes.

    Under the hood, `'categorical'` dtype is a representation that maps each
    category to an integer, thus providing a more memory-efficient object
    structure than, e.g., 'str', and allowing faster grouping, mapping, and similar
    operations on the resulting object.

    MatchCategories() remembers the encodings or levels that represent each
    category, and can thus can be used to ensure that the correct encoding gets
    applied when passing categorical data to modeling packages that support this
    dtype, or to prevent unseen categories from reaching a further transformer
    or estimator in a pipeline, for example.

    More details in the :ref:`User Guide <match_categories>`.

    Parameters
    ----------
    {variables}

    {ignore_format}

    {missing_values}

    Attributes
    ----------
    category_dict_:
        Dictionary with the category encodings assigned to each variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the encodings or levels to use for each variable.

    fit_transform:
        Fit to the data. Then transform it.

    get_feature_names_out:
        Get output feature names for transformation.

    get_params:
        Get parameters for this estimator.

    set_params:
        Set the parameters of this estimator.

    transform:
        Enforce the type of categorical variables as dtype `categorical`.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.preprocessing import MatchCategories
    >>> X_train = pd.DataFrame(dict(x1 = ["a","b","c"], x2 = [4,5,6]))
    >>> X_test = pd.DataFrame(dict(x1 = ["c","b","a","d"], x2 = [5,6,4,7]))
    >>> mc = MatchCategories(missing_values="ignore")
    >>> mc.fit(X_train)
    >>> mc.transform(X_train)
      x1  x2
    0  a   4
    1  b   5
    2  c   6
    >>> mc.transform(X_test)
        x1  x2
    0    c   5
    1    b   6
    2    a   4
    3  NaN   7
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
        missing_values: str = "raise",
    ) -> None:

        super().__init__(variables, missing_values, ignore_format)

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
        X = check_X(X)
        variables_ = self._check_or_select_variables(X)

        if self.missing_values == "raise":
            _check_optional_contains_na(X, variables_)

        self.category_dict_ = dict()
        for var in variables_:
            self.category_dict_[var] = pd.Categorical(X[var]).categories

        self.variables_ = variables_
        self._get_feature_names_in(X)
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
            The dataframe with the variables encoded as pandas categorical dtype.
        """
        X = self._check_transform_input_and_state(X)

        if self.missing_values == "raise":
            _check_optional_contains_na(X, self.variables_)

        for feature, levels in self.category_dict_.items():
            X[feature] = pd.Categorical(X[feature], levels)

        self._check_nas_in_result(X)
        return X

    def _check_nas_in_result(self, X: pd.DataFrame):
        # check if NaN values were introduced by the encoding
        if X[self.category_dict_.keys()].isnull().sum().sum() > 0:

            # obtain the name(s) of the columns that have null values
            nan_columns = (
                X[self.category_dict_.keys()]
                .columns[X[self.category_dict_.keys()].isnull().any()]
                .tolist()
            )

            if len(nan_columns) > 1:
                nan_columns_str = ", ".join(nan_columns)
            else:
                nan_columns_str = nan_columns[0]

            if self.missing_values == "ignore":
                warnings.warn(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )
            elif self.missing_values == "raise":
                raise ValueError(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )
