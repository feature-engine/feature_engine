# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters import (
    _ignore_format_docstring,
    _unseen_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
    _transform_encoders_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X_y
from feature_engine.encoding._helper_functions import check_parameter_unseen
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixin,
    CategoricalMethodsMixin,
)
from feature_engine.tags import _return_tags


class WoE:
    def _check_fit_input(self, X: pd.DataFrame, y: pd.Series):
        """
        Check that X is dataframe, and y a binary series with values 0 and 1.
        """
        X, y = check_X_y(X, y)

        # check that y is binary
        if y.nunique() != 2:
            raise ValueError(
                "This encoder is designed for binary classification. The target "
                "used has more than 2 unique values."
            )

        # if target does not have values 0 and 1, we need to remap, to be able to
        # compute the averages.
        if y.min() != 0 or y.max() != 1:
            y = pd.Series(np.where(y == y.min(), 0, 1))
        return X, y

    def _calculate_woe(self, X: pd.DataFrame, y: pd.Series, variable: Union[str, int]):
        total_pos = y.sum()
        inverse_y = y.ne(1).copy()
        total_neg = inverse_y.sum()

        pos = y.groupby(X[variable]).sum() / total_pos
        neg = inverse_y.groupby(X[variable]).sum() / total_neg

        if not (pos[:] == 0).sum() == 0 or not (neg[:] == 0).sum() == 0:
            raise ValueError(
                "The proportion of one of the classes for a category in "
                "variable {} is zero, and log of zero is not defined".format(variable)
            )

        woe = np.log(pos / neg)
        return pos, neg, woe


@Substitution(
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
class WoEEncoder(CategoricalInitMixin, CategoricalMethodsMixin, WoE):
    """
    The WoEEncoder() replaces categories by the weight of evidence
    (WoE). The WoE was used primarily in the financial sector to create credit risk
    scorecards.

    The encoder will encode only categorical variables by default
    (type 'object' or 'categorical'). You can pass a list of variables to encode.
    Alternatively, the encoder will find and encode all categorical variables
    (type 'object' or 'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the weight of evidence for each variable
    (fit). The encoder then transforms the categories into the mapped numbers
    (transform).

    This categorical encoding is exclusive for binary classification.

    **Note**

    The log(0) is not defined and the division by 0 is not defined. Thus, if any of the
    terms in the WoE equation are 0 for a given category, the encoder will return an
    error. If this happens, try grouping less frequent categories.

    More details in the :ref:`User Guide <woe_encoder>`.

    Parameters
    ----------
    {variables}

    {ignore_format}

    {unseen}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the WoE per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the WoE per category, per variable.

    {transform}

    {fit_transform}

    {inverse_transform}

    Notes
    -----
    For details on the calculation of the weight of evidence visit:
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    There is a similar implementation in the the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    feature_engine.discretisation
    category_encoders.woe.WOEEncoder
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
        unseen: str = "ignore",
    ) -> None:

        super().__init__(variables, ignore_format)
        check_parameter_unseen(unseen, ["ignore", "raise"])
        self.unseen = unseen

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the WoE.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y: pandas series.
            Target, must be binary.
        """

        X, y = self._check_fit_input(X, y)

        self._fit(X)
        self._get_feature_names_in(X)

        self.encoder_dict_ = {}

        for var in self.variables_:
            _, _, woe = self._calculate_woe(X, y, var)

            self.encoder_dict_[var] = woe.to_dict()

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        tags_dict["requires_y"] = True
        # in the current format, the tests are performed using continuous np.arrays
        # this means that when we encode some of the values, the denominator is 0
        # and this the transformer raises an error, and the test fails.
        # For this reason, most sklearn transformers will fail. And it has nothing to
        # do with the class not being compatible, it is just that the inputs passed
        # are not suitable
        tags_dict["_skip_test"] = True
        return tags_dict
