# Authors: Ankit Hemant Lade (contributor)
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine.dataframe_checks import (
    _check_optional_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_all_variables,
    find_all_variables,
)


class StringListBinarizer(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    StringListBinarizer() takes categorical variables that contain a list of strings
    or a delimited string, and creates binary variables representing each of the
    unique categories across all observations.

    This is especially useful for columns containing multiple tags per row, such as
    `["action", "comedy"]` or `"action, comedy"`.

    The transformer takes a list of variables to encode, or automatically selects
    all object/categorical columns if none are provided.

    The encodings are created by splitting the strings on a specified `separator`
    (or parsing the lists directly), identifying the unique tags in the dataset,
    and then adding a new boolean column `varname_tag` for each unique tag.

    Original columns are dropped after transformation by default.

    More details in the :ref:`User Guide <string_list_binarizer>`.

    Parameters
    ----------
    variables : list, default=None
        The list of categorical variables to encode. If None, the encoder will find and
        select all categorical variables.

    separator : str, default=","
        The separator used to split the strings in the variable.
        If the variable contains Python lists instead of strings,
        this parameter is ignored.

    ignore_format : bool, default=False
        Whether to format check the variables in `fit`. If `True`, the encoder will
        ignore the variable types and proceed with encoding, provided the variables are
        entered by the user. If `variables` is None, the target variables are all those
        in the dataset regardless of type. If `False`, the encoder will select and
        encode only categorical variables (type 'object' or 'categorical').

    Attributes
    ----------
    variables_:
        The list of variables to be transformed.

    encoder_dict_:
        A dictionary mapping the variables to the sorted list of their unique tags.

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Learn the unique tags per variable.

    fit_transform:
        Fit to data, then transform it.

    transform:
        Replace the original variable with the binary encoded variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.encoding import StringListBinarizer
    >>> X = pd.DataFrame(dict(tags=["action, comedy", "comedy", "action, thriller"]))
    >>> slb = StringListBinarizer(variables=["tags"], separator=", ")
    >>> slb.fit(X)
    >>> slb.transform(X)
       tags_action  tags_comedy  tags_thriller
    0            1            1              0
    1            0            1              0
    2            1            0              1
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        separator: str = ",",
        ignore_format: bool = False,
    ) -> None:

        if not isinstance(separator, str):
            raise ValueError(
                f"separator takes only strings. Got {type(separator).__name__} instead."
            )

        if not isinstance(ignore_format, bool):
            raise ValueError(
                "ignore_format takes only booleans True and False. "
                f"Got {ignore_format} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.separator = separator
        self.ignore_format = ignore_format

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the unique tags present in each categorical variable.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas series, default=None
            Target. It is not needed in this encoded. You can pass y or
            None.
        """
        X = check_X(X)

        # select variables to encode
        if self.ignore_format is True:
            if self.variables is None:
                self.variables_ = find_all_variables(X)
            else:
                self.variables_ = check_all_variables(X, self.variables)
        else:
            if self.variables is None:
                # Select typical categorical/string-like variables
                self.variables_ = X.select_dtypes(
                    include=["object", "category", "string"]
                ).columns.to_list()
                if len(self.variables_) == 0:
                    raise ValueError(
                        "No categorical variables found in the dataframe. Please check "
                        "the variables format or set `ignore_format=True`."
                    )
            else:
                self.variables_ = _check_variables_input_value(self.variables)

                # Check that specified variables exist and are object/categorical
                non_cat = [
                    var
                    for var in self.variables_
                    if var in X.columns
                    and not (
                        pd.api.types.is_object_dtype(X[var])
                        or isinstance(X[var].dtype, pd.CategoricalDtype)
                        or pd.api.types.is_string_dtype(X[var])
                    )
                ]
                if non_cat:
                    raise TypeError(
                        "Some of the variables are not categorical. Please cast them "
                        "as object or categorical before calling fit, or set "
                        "`ignore_format=True`. Variables: "
                        f"{non_cat}"
                    )

        _check_optional_contains_na(X, self.variables_)

        self.encoder_dict_ = {}

        for var in self.variables_:
            unique_tags = set()
            for row in X[var]:
                if isinstance(row, str):
                    tags = [t.strip() for t in row.split(self.separator)]
                elif isinstance(row, list):
                    tags = [str(t).strip() for t in row]
                else:
                    tags = [str(row).strip()]
                unique_tags.update(tags)

            # Remove empty strings from tags (often caused by trailing separators)
            unique_tags.discard("")

            self.encoder_dict_[var] = sorted(list(unique_tags))

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace the categorical variables by the binary encoded variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe.
            The transformed dataframe. The shape of the dataframe will differ from
            the original, as it replaces the original list/string columns with multiple
            dummy columns.
        """
        check_is_fitted(self)

        X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)
        _check_optional_contains_na(X, self.variables_)

        X_transformed = X[self.feature_names_in_].copy()

        for feature in self.variables_:
            categories = self.encoder_dict_[feature]

            # Use faster numpy processing for dummies
            dummy_data = {
                f"{feature}_{category}": np.zeros(len(X), dtype=int)
                for category in categories
            }

            for i, row in enumerate(X[feature]):
                if isinstance(row, str):
                    tags = [t.strip() for t in row.split(self.separator)]
                elif isinstance(row, list):
                    tags = [str(t).strip() for t in row]
                else:
                    tags = [str(row).strip()]

                for t in tags:
                    if t in categories:
                        dummy_data[f"{feature}_{t}"][i] = 1

            dummy_df = pd.DataFrame(dummy_data, index=X.index)
            X_transformed = pd.concat([X_transformed, dummy_df], axis=1)

        # drop original variables
        X_transformed.drop(labels=self.variables_, axis=1, inplace=True)

        return X_transformed

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Get output feature names for transformation."""
        check_is_fitted(self)

        feature_names = list(self.feature_names_in_)
        feature_names = [f for f in feature_names if f not in self.variables_]

        for feature in self.variables_:
            for category in self.encoder_dict_[feature]:
                feature_names.append(f"{feature}_{category}")

        return feature_names

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict
