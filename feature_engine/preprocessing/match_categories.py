import warnings
from typing import List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _missing_values_docstring,
    _return_empty_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.init_parameters.encoders import _ignore_format_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixinNA,
    CategoricalMethodsMixin,
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    missing_values=_missing_values_docstring,
    variables=_variables_categorical_docstring,
    return_empty=_return_empty_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
)
class MatchCategories(
    CategoricalMethodsMixin, CategoricalInitMixinNA, GetFeatureNamesOutMixin
):
    """
    MatchCategories() ensures that categorical variables are encoded as pandas
    `'categorical'` dtype, or polars `'Enum'` dtype, instead of generic python
    `'object'`, string or other dtypes.

    Under the hood, `'categorical'` dtype is a representation that maps each
    category to an integer, thus providing a more memory-efficient object
    structure than, e.g., 'str', and allowing faster grouping, mapping, and similar
    operations on the resulting object.

    MatchCategories() remembers the encodings or levels that represent each
    category, and can thus be used to ensure that the correct encoding gets
    applied when passing categorical data to modelling packages that support this
    dtype, or to prevent unseen categories from reaching a further transformer
    or estimator in a pipeline, for example. Categories not seen during fit become
    missing values.

    The polars `'Enum'` dtype only takes strings, so with polars, numerical
    variables cast with `ignore_format=True` become strings.

    More details in the :ref:`User Guide <match_categories>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    {ignore_format}

    {missing_values}

    Attributes
    ----------
    category_dict_:
        Dictionary with the categories learned for each variable. With polars, the
        categories are stored as lists of strings.

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
        Cast the categorical variables to a categorical dtype.

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
        return_empty: bool = False,
        ignore_format: bool = False,
        missing_values: str = "raise",
    ) -> None:

        _check_return_empty_is_bool(return_empty)

        super().__init__(variables, missing_values, ignore_format)
        self.return_empty = return_empty

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Learn the categories of each categorical variable.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: Series, default = None
            y is not needed in this transformer. You can pass y or None.
        """
        nw_X = check_X(X)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        if nwd.is_pandas_dataframe(X) is True:
            # pandas is faster than narwhals.
            self.category_dict_ = {
                var: X[var].astype("category").cat.categories for var in variables_
            }
        else:
            self.category_dict_ = {
                var: self._find_categories(nw_X.get_column(var)) for var in variables_
            }

        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Cast the categorical variables to a categorical dtype with the categories
        learned during fit. Categories not seen during fit become missing values.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features].
            The dataset to transform.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features].
            The dataframe with the variables cast to pandas `category` or polars
            `Enum` dtype.
        """
        nw_X = self._check_transform_input_and_state(X)
        self._check_na(X, self.variables_)

        if nwd.is_pandas_dataframe(X) is True:
            # pandas is faster than narwhals.
            X = X.copy()
            categorical = nw.get_native_namespace(nw_X).Categorical
            for feature, levels in self.category_dict_.items():
                # get_indexer returns -1 for unseen categories, which from_codes
                # turns into NaN.
                X[feature] = categorical.from_codes(
                    levels.get_indexer(X[feature]), categories=levels
                )
            nw_X = nw.from_native(X, eager_only=True)
        else:
            nw_X = nw_X.with_columns(
                *[
                    nw.when(nw.col(feature).cast(nw.String).is_in(levels))
                    .then(nw.col(feature).cast(nw.String))
                    .cast(nw.Enum(levels))
                    for feature, levels in self.category_dict_.items()
                ]
            )

        self._check_nas_in_result(nw_X)
        return nw_X.to_native()

    def _check_nas_in_result(self, nw_X: nw.DataFrame):
        nan_columns = [
            str(feature)
            for feature in self.category_dict_
            if nw_X.get_column(feature).null_count() > 0
        ]

        if len(nan_columns) > 0:
            msg = (
                "During the encoding, NaN values were introduced in the feature(s) "
                f"{', '.join(nan_columns)}."
            )
            if self.missing_values == "ignore":
                warnings.warn(msg)
            elif self.missing_values == "raise":
                raise ValueError(msg)

    def _find_categories(self, series: nw.Series) -> List[str]:
        if series.dtype == nw.Enum:
            return list(series.dtype.categories)
        if series.dtype.is_float() is True:
            # NaN is a missing value in pandas, but a regular value in polars.
            series = series.filter(~series.is_nan())
        # polars Enum only takes strings, so categories are sorted in the original
        # dtype, to keep numbers in numeric order, and then cast to string.
        return series.drop_nulls().unique().sort().cast(nw.String).to_list()
