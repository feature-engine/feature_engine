from typing import List, Optional, Union

import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _missing_values_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.init_parameters.encoders import (
    _ignore_format_docstring,
    _unseen_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
    _transform_encoders_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.encoding._helper_functions import check_parameter_unseen
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixinNA,
    CategoricalMethodsMixin,
)
from feature_engine.dataframe_checks import (
    _check_optional_contains_na,
)


@Substitution(
    missing_values=_missing_values_docstring,
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
    unseen=_unseen_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    transform=_transform_encoders_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class PandasCategoricalEncoder(CategoricalInitMixinNA, CategoricalMethodsMixin):
    """Transform columns into pandas categorical type columns.

    Simply applying pandas.to_categorical() separately on train and test set
    will not guarantee that each category are encoded in the same way in both datasets.

    This class addresses this problem by making sure that categories are encoded
    consistently between train and test set.

    When `unseen="ignore"` unseen categories encountered during transform are
    transformed to NAN when the unseen parameter and will have an associated encoded
    value of -1.

    Parameters
    ----------

    {variables}

    {missing_values}

    {ignore_format}

    {unseen}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the ordinal number per category, per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find the integer to replace each category in each variable.

    {fit_transform}

    {inverse_transform}

    {transform}

    Notes
    -----
    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    category_encoders.ordinal.OrdinalEncoder

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.encoding import PandasCategoricalEncoder
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4], x2 = ["c", "a", "b", "c"]))
    >>> y = pd.Series([0,1,1,0])
    >>> pandas_cat_encoder = PandasCategoricalEncoder()
    >>> pandas_cat_encoder.fit(X)
    >>> X_transformed = pandas_cat_encoder.transform(X)
    >>> X_transformed
    x1 x2
    0   1  c
    1   2  a
    2   3  b
    3   4  c
    >>> X_transformed.dtypes
    x1       int64
    x2    category
    dtype: object
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
        ignore_format: bool = False,
        unseen: str = "ignore",
    ) -> None:

        check_parameter_unseen(unseen, ["ignore", "raise"])
        super().__init__(variables, missing_values, ignore_format)
        self.unseen = unseen

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learn the numbers to be used to replace the categories in each
        variable.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to be encoded.

        y: pandas series, default=None
            The Target. Can be None if `encoding_method='arbitrary'`.
            Otherwise, y needs to be passed when fitting the transformer.
        """

        X = check_X(X)

        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}
        for feature in variables_:
            self.encoder_dict_[feature] = {
                category: index
                for index, category in enumerate(
                    sorted([val for val in X[feature].unique() if pd.notnull(val)])
                )
            }

        if self.unseen == "encode":
            self._unseen = -1

        # assign underscore parameters at the end in case code above fails
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def transform(self, X):
        """
        Transforms the specified columns in the DataFrame to categorical dtype.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with specified columns converted to categorical
                dtype.
        """
        X = self._check_transform_input_and_state(X)
        # check if dataset contains na
        if self.missing_values == "raise":
            _check_optional_contains_na(X, self.variables_)

        for feature in self.variables:
            X[feature] = pd.Categorical(
                X[feature],
                # categories are sorted to ensure consistency between train and test set
                categories=sorted(
                    self.encoder_dict_[feature], key=self.encoder_dict_[feature].get
                ),
            )

        if self.unseen == "raise":
            self._check_nan_values_after_transformation(X)

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert the encoded variable back to the original values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------
        X_tr: pandas dataframe of shape = [n_samples, n_features].
            The un-transformed dataframe, with the categorical variables containing the
            original values.
        """
        X = self._check_transform_input_and_state(X)

        # replace encoded categories by the original values
        for feature in self.encoder_dict_.keys():
            inv_map = {v: k for k, v in self.encoder_dict_[feature].items()}
            X[feature] = X[feature].cat.codes.map(inv_map)

        return X
