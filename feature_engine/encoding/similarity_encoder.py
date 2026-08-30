from difflib import SequenceMatcher
from typing import List, Optional, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.utils.validation import check_is_fitted

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.init_parameters.encoders import _ignore_format_docstring
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import _check_contains_na, check_X
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixin,
    CategoricalMethodsMixin,
)


def _gpm_fast(x1: str, x2: str) -> float:
    return SequenceMatcher(None, str(x1), str(x2)).quick_ratio()


_gpm_fast_vec = np.vectorize(_gpm_fast)


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
    return_empty=_return_empty_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class StringSimilarityEncoder(CategoricalMethodsMixin, CategoricalInitMixin):
    """
    The StringSimilarityEncoder() replaces categorical variables with a set of float
    variables that capture the similarity between the category names. The new variables
    have values between 0 and 1, where 0 indicates no similarity and 1 is an exact
    match between the names of the categories.

    The similarity measure is a float in the range [0, 1]. It is defined as 2 * M / T,
    where T is the total number of elements in both categories being compared, and M is
    the number of matches. Note that this is 1 if the sequences are identical, and 0 if
    they have nothing in common.

    For example, the similarity between the categories "dog" and "dig" is 0.66. T is the
    total number of elements in both categories, that is 6. There are 2 matches between
    the words, the letters d and g, so: 2 * M / T = 2 * 2 / 6 = 0.66.

    This encoding is similar to one-hot encoding, in the sense that each category is
    encoded as a new variable. But the values, instead of 1 or 0, are the similarity
    between the observation's category and the dummy variable.

    For example, if a variable has 3 categories, dog, dig and cat,
    StringSimilarityEncoder() will create 3 new variables, var_dog, var_dig and var_cat
    and the values would be for the observation dog: 1, 0.66 , 0. For the observation
    dig they would be 0.66, 1, 0. And for cat, they would be 0, 0, 1.

    The encoder has the option to generate similarity variables only for the most
    popular categories, that is, the categories present in most observations. This
    behaviour can be specified with the parameter `top_categories`.

    **Missing values**

    StringSimilarityEncoder() will replace missing data with an empty string and
    then return the similarity to the remaining variables by default. Alternatively,
    it can be set to return an error if the variable has missing values, or to ignore
    them.

    **Unseen categories**

    StringSimilarityEncoder() handles unseen categories out-of-the-box by assigning a
    similarity measure to the other categories that were seen during `fit()`.

    **Categorical variables**

    The encoder will encode only categorical variables by default (type 'object' or
    'categorical'). You can pass a list of variables to encode. Alternatively, the
    encoder will find and encode all categorical variables.

    **Numerical variables**

    With `ignore_format=True` you have the option to encode numerical variables as well.
    Encoding numerical variables with similarity measures make sense for example for
    variables like barcodes. In this case, you can either enter the list of variables
    to encode (recommended), or the transformer will automatically select all variables.

    More details in the :ref:`User Guide <string_similarity>`.

    Parameters
    ----------
    top_categories: int, default=None
        If None, dummy variables will be created for each unique category of the
        variable. Alternatively, we can indicate in the number of most frequent
        categories to encode. In this case, similarity variables will be created
        only for those popular categories.

    missing_values: str, default='impute'
        Indicates if missing values should be ignored, raised or imputed. If 'raise' the
        transformer will return an error if the datasets to `fit` or `transform`
        contain missing values. If 'ignore', missing data will be ignored when learning
        parameters or performing the transformation. If 'impute', the transformer will
        replace missing values with an empty string, '', and then return the similarity
        measures.

    keywords: dict, default=None
        Dictionary with a set of keywords to be used to create the similarity variables.
        The format should be: dict(feature: [keyword1, keyword2, ...]). The encoder will
        use these keywords to create the similarity variables. The dictionary can be
        defined for all the features to encode, or only for a subset of them. In this
        case, for the features not specified in the dictionary, the encoder will
        identify the categories from the data.

    {variables}

    {return_empty}

    {ignore_format}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the categories for which dummy variables will be created.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the unique categories per variable.

    {fit_transform}

    transform:
        Replace the categorical variables by the distance variables.

    Notes
    -----
    This encoder will encode unseen categories by measuring string similarity between
    seen and unseen categories.

    No text preprocessing is applied before calculating the similarity.

    The original categorical variables are removed from the returned dataset after the
    transformation. In their place, the binary variables are returned.

    See Also
    --------
    feature_engine.encoding.OneHotEncoder
    dirty_cat.SimilarityEncoder

    References
    ----------
    .. [1] Cerda P, Varoquaux G, Kégl B. "Similarity encoding for learning with dirty
       categorical variables". Machine Learning, Springer Verlag, 2018.
    .. [2] Cerda P, Varoquaux G. "Encoding high-cardinality string categorical
       variables". IEEE Transactions on Knowledge & Data Engineering, 2020.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.encoding import StringSimilarityEncoder
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4], x2 = ["dog", "dig", "dagger", "hi"]))
    >>> sse = StringSimilarityEncoder()
    >>> sse.fit(X)
    >>> sse.transform(X)
       x1    x2_dog    x2_dig  x2_dagger  x2_hi
    0   1  1.000000  0.666667   0.444444    0.0
    1   2  0.666667  1.000000   0.444444    0.4
    2   3  0.444444  0.444444   1.000000    0.0
    3   4  0.000000  0.400000   0.000000    1.0

    With polars

    >>> import polars as pl
    >>> from feature_engine.encoding import StringSimilarityEncoder
    >>> X = pl.DataFrame(dict(x1 = [1,2,3,4], x2 = ["dog", "dig", "dagger", "hi"]))
    >>> sse = StringSimilarityEncoder()
    >>> sse.fit(X)
    >>> sse.transform(X)
    shape: (4, 5)
    ┌─────┬──────────┬──────────┬───────────┬───────┐
    │ x1  ┆ x2_dog   ┆ x2_dig   ┆ x2_dagger ┆ x2_hi │
    │ --- ┆ ---      ┆ ---      ┆ ---       ┆ ---   │
    │ i64 ┆ f64      ┆ f64      ┆ f64       ┆ f64   │
    ╞═════╪══════════╪══════════╪═══════════╪═══════╡
    │ 1   ┆ 1.0      ┆ 0.666667 ┆ 0.444444  ┆ 0.0   │
    │ 2   ┆ 0.666667 ┆ 1.0      ┆ 0.444444  ┆ 0.4   │
    │ 3   ┆ 0.444444 ┆ 0.444444 ┆ 1.0       ┆ 0.0   │
    │ 4   ┆ 0.0      ┆ 0.4      ┆ 0.0       ┆ 1.0   │
    └─────┴──────────┴──────────┴───────────┴───────┘
    """

    def __init__(
        self,
        top_categories: Optional[int] = None,
        keywords: Optional[dict] = None,
        missing_values: str = "impute",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        ignore_format: bool = False,
    ):
        if top_categories and not isinstance(top_categories, int):
            raise ValueError(
                f"top_categories takes only integers. Got {top_categories!r} instead."
            )
        if missing_values not in ("raise", "impute", "ignore"):
            raise ValueError(
                "missing_values should be one of 'raise', 'impute' or 'ignore'."
                f" Got {missing_values!r} instead."
            )
        if keywords and not isinstance(keywords, dict):
            raise ValueError(
                f"keywords should be a dictionary or None. Got {keywords!r} instead."
            )
        if keywords and not all(isinstance(item, list) for item in keywords.values()):
            raise ValueError(
                "The items in keywords should be lists."
                f" Got {keywords.values()!r} instead."
            )
        super().__init__(variables, return_empty, ignore_format)
        self.top_categories = top_categories
        self.missing_values = missing_values
        self.keywords = keywords

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Learns the unique categories per variable. If top_categories is indicated,
        it will learn the most popular categories. Alternatively, it learns all
        unique categories per variable.

        Parameters
        ----------

        X: dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to encode.

        y: Series, default=None
            Target. It is not needed in this encoder. You can pass y or None.
        """

        nw_X = check_X(X)
        variables_ = self._check_or_select_variables(X)

        if self.keywords and not all(
            item in variables_ for item in self.keywords.keys()
        ):
            raise ValueError(
                "There are variables in keywords that are not present "
                "in the dataset."
            )

        # if data contains nan, fail before running any logic
        if self.missing_values == "raise":
            _check_contains_na(X, variables_, error_msg="optional")

        self.encoder_dict_ = {}

        if self.keywords:
            self.encoder_dict_.update(self.keywords)
            cols_to_iterate = [x for x in variables_ if x not in self.keywords]
        else:
            cols_to_iterate = variables_

        # cast(nw.String) preserves nulls as null on both backends (unlike
        # pandas' own astype(str), which stringifies NaN to "nan"), so
        # "impute" can fill_null("") directly and "ignore" can drop_nulls()
        # before casting, with no leftover "nan"/"<NA>" text sentinels to
        # special-case downstream.
        for var in cols_to_iterate:
            col = nw_X.get_column(var)
            if self.missing_values == "impute":
                col = col.cast(nw.String).fill_null("")
            elif self.missing_values == "ignore":
                col = col.drop_nulls().cast(nw.String)
            elif self.missing_values == "raise":
                col = col.cast(nw.String)
            else:
                # missing_values can be set directly (e.g. via set_params or
                # attribute assignment) bypassing the __init__ check above.
                raise ValueError(
                    "Unrecognized value for missing_values. It should be 'raise', "
                    f"'ignore' or 'impute'. Got {self.missing_values} instead."
                )

            # sort=True mirrors pandas' own value_counts() default order
            # (descending by count, ties broken by first appearance), so
            # encoder_dict_ keeps the same category order as before.
            counts = col.value_counts(sort=True)
            categories = counts.get_column(counts.columns[0]).to_list()
            self.encoder_dict_[var] = categories[: self.top_categories]

        # assign underscore parameters at the end in case code above fails
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Replaces the categorical variables with the similarity variables.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe.
            The transformed dataframe. The shape of the dataframe will be different from
            the original as it includes the similarity variables in place of the
            original categorical ones.
        """

        check_is_fitted(self)
        nw_X = self._check_transform_input_and_state(X)
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_, error_msg="optional")

        if len(self.variables_) == 0:
            return nw_X.to_native()

        # String similarity (difflib.SequenceMatcher) has no vectorised
        # narwhals/backend equivalent, so it is computed in numpy: the
        # similarity matrix is built once per unique value (not per row)
        # via broadcasting, then broadcast back to all rows with
        # np.unique's inverse index - cheaper than a per-row Python-level
        # dict lookup and it is backend agnostic, so a single code path
        # covers both pandas and polars. Benchmarked at 10k-100k rows x
        # 1-10 columns x 5-50 categories: the difflib computation itself
        # dominates wall time by 1-3 orders of magnitude, so a
        # pandas-specific fast path for reassembling the output columns
        # (as used in DecisionTreeFeatures) would save at most ~10ms out of
        # a transform that is already tens of ms to seconds - not worth the
        # code duplication here.
        new_series = []
        for var in self.variables_:
            col = nw_X.get_column(var)
            categories = self.encoder_dict_[var]

            null_mask = None
            if self.missing_values == "impute":
                str_col = col.cast(nw.String).fill_null("")
            else:
                str_col = col.cast(nw.String)
                if self.missing_values == "ignore":
                    null_mask = np.array(col.is_null().to_list())

            values = np.asarray(str_col.to_list(), dtype=object)
            if null_mask is not None:
                # placeholder value for null rows: overwritten with NaN
                # below, the string itself is never used.
                values = np.where(null_mask, "", values)

            unique_vals, inverse = np.unique(values, return_inverse=True)
            cats_arr = np.asarray(categories, dtype=object)
            sim_matrix = _gpm_fast_vec(
                unique_vals.reshape(-1, 1), cats_arr.reshape(1, -1)
            )
            encoded = sim_matrix[inverse]

            if null_mask is not None:
                encoded[null_mask, :] = np.nan

            for j, category in enumerate(categories):
                name = f"{var}_nan" if category == "" else f"{var}_{category}"
                new_series.append(
                    nw.new_series(name, encoded[:, j], backend=nw_X.implementation)
                )

        nw_X = nw_X.with_columns(*new_series).drop(self.variables_)

        return nw_X.to_native()

    def _get_new_features_name(self) -> List[str]:
        """Return names of the created features."""
        feature_names = []
        for feature in self.variables_:
            for category in self.encoder_dict_[feature]:
                if category == "":
                    feature_names.append(f"{feature}_nan")
                else:
                    feature_names.append(f"{feature}_{category}")

        return feature_names

    def _add_new_feature_names(self, feature_names: List[str]) -> List[str]:
        """Creates new features names and removes original categorical variables."""
        feature_names = feature_names + self._get_new_features_name()
        feature_names = [f for f in feature_names if f not in self.variables_]

        return feature_names

    def inverse_transform(self, X: IntoDataFrame):
        """inverse_transform is not implemented for this transformer."""
        raise NotImplementedError(
            "inverse_transform is not implemented for this transformer."
        )
