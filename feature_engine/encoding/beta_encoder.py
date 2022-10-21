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
from feature_engine.encoding._helper_functions import check_parameter_unseen
from feature_engine.dataframe_checks import check_X_y
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixin,
    CategoricalMethodsMixin,
)
from feature_engine.tags import _return_tags


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
class BetaEncoder(CategoricalInitMixin, CategoricalMethodsMixin):
    """
    The BetaEncoder() replaces categories by cantral tendency (statistic parameter)
    of the beta distribution modeled from the data.

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

    More details in the :ref:`User Guide <beta_encoder>`.

    Parameters
    ----------
    statistic: str, default = "mean"
        Could be one of {'mean', 'median', 'mode', 'geomean', 'harmmean'}. Chosen statistic
        will be calculate during fit for each category in the column.
        
    {variables}

    {ignore_format}

    {unseen}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the beta distribution statistic per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the beta distribution statistic per category, per variable.

    {transform}

    {fit_transform}

    {inverse_transform}

    Notes
    -----
    For more examples of Beta Encoding please visit:
    https://www.kaggle.com/code/mmotoki/beta-target-encoding/notebook
    https://towardsdatascience.com/target-encoding-and-bayesian-target-encoding-5c6a6c58ae8c
    https://rdrr.io/github/mattmotoki/kaggleUtils/man/betaEncoder.html
    https://en.wikipedia.org/wiki/Beta_distribution

    See Also
    --------
    feature_engine.encoding.MeanEncoder
    category_encoders.woe.WOEEncoder
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
        unseen: str = "ignore",
        statistic: str = "mean",
    ) -> None:

        super().__init__(variables, ignore_format)
        check_parameter_unseen(unseen, ["ignore", "raise", "encode"])
        self.unseen = unseen
        self.statistic = statistic

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
            y = y.gt(y.mean())

        self._fit(X)
        self._get_feature_names_in(X)

        self.encoder_dict_ = {}

        prior_alpha = np.sum(y)
        prior_beta = len(y) - prior_alpha
        if self.unseen == "encode":
            self._unseen = self._calc_central_tendency(
                prior_alpha,
                prior_beta,
                self.statistic
            )

        for var in self.variables_:
            stats = y.groupby(X[var]).agg(["sum", "count"])
            # updating number of 'wins'
            posterior_alpha = prior_alpha + stats["sum"]
            # updating number of 'losses'
            posterior_beta = prior_beta + (stats["count"] - stats["sum"])
            # rescaling back to original scale
            self.encoder_dict_[var] = self._calc_central_tendency(
                posterior_alpha,
                posterior_beta,
                self.statistic
            ).to_dict()

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

    def _calc_central_tendency(self, alpha, beta, encode):
        """
        Helper function to calculate different central tendencies
        based on beta distribution parameters
        Parameters
        ----------
        alpha : float
            Alpha parameter of beta distribution.
        beta : float
            Beta parameter of beta distribution.
        encode : {'mean', 'median', 'mode', 'geomean', 'harmmean'}
            Central tendency to calculate.
        """
        if encode == "mean":
            return alpha / (alpha + beta)
        elif encode == "median":
            return (alpha - 1./3.) / (alpha + beta - 2./3.)
        elif encode == "mode":
            return (alpha - 1.) / (alpha + beta - 2.)
        elif encode == "geomean":
            return (alpha - 0.5) / (alpha + beta - 0.5)
        elif encode == "harmmean":
            return (alpha - 1.) / (alpha + beta - 1.)
        else:
            raise ValueError("Wrong encoding option")
