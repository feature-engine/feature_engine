# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class WoEEncoder(BaseCategoricalTransformer):
    """
    The WoERatioCategoricalEncoder() replaces categories by the weight of evidence
    (WoE). The WoE was used primarily in the financial sector to create credit risk
    scorecards.

    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed the encoder
    will find and encode all categorical variables (object type).

    The encoder first maps the categories to the weight of evidence for each variable
    (fit). The encoder then transforms the categories into the mapped numbers
    (transform).

    **Note**

    This categorical encoding is exclusive for binary classification.

    **The weight of evidence is given by:**

    .. math::

        log( p(X=xj|Y = 1) / p(X=xj|Y=0) )



    **The WoE is determined as follows:**

    We calculate the percentage positive cases in each category of the total of all
    positive cases. For example 20 positive cases in category A out of 100 total
    positive cases equals 20 %. Next, we calculate the percentage of negative cases in
    each category respect to the total negative cases, for example 5 negative cases in
    category A out of a total of 50 negative cases equals 10%. Then we calculate the
    WoE by dividing the category percentages of positive cases by the category
    percentage of negative cases, and take the logarithm, so for category A in our
    example WoE = log(20/10).

    **Note**

    - If WoE values are negative, negative cases supersede the positive cases.
    - If WoE values are positive, positive cases supersede the negative cases.
    - And if WoE is 0, then there are equal number of positive and negative examples.

    **Encoding into WoE**:

    - Creates a monotonic relationship between the encoded variable and the target
    - Returns variables in a similar scale

    **Note**

    The log(0) is not defined and the division by 0 is not defined. Thus, if any of the
    terms in the WoE equation are 0 for a given category, the encoder will return an
    error. If this happens, try grouping less frequent categories.

    Parameters
    ----------
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and select all object type variables.

    Attributes
    ----------
    encoder_dict_ :
        Dictionary with the WoE per variable.

    Methods
    -------
    fit:
        Learn the WoE per category, per variable.
    transform:
        Encode the categories to numbers.
    fit_transform:
        Fit to the data, then transform it.
    inverse_transform:
        Encode the numbers into the original categories.

    Notes
    -----
    For details on the calculation of the weight of evidence visit:
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    In credit scoring, continuous variables are also transformed using the WoE. To do
    this, first variables are sorted into a discrete number of bins, and then these
    bins are encoded with the WoE as explained here for categorical variables. You can
    do this by combining the use of the equal width, equal frequency or arbitrary
    discretisers.

    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    feature_engine.discretisation
    """

    def __init__(
        self, variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:

        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the the WoE.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y : pandas series.
            Target, must be binary [0,1].

        Raises
        ------
        TypeError
            - If the input is not the Pandas DataFrame.
            - If any user provided variables are not categorical.
        ValueError
            - If there are no categorical variables in df or df is empty
            - If variable(s) contain null values.
            - If y is not binary with values 0 and 1.
            - If p(0) = 0 or p(1) = 0.

        Returns
        -------
        self
        """

        X = self._check_fit_input_and_variables(X)

        # check that y is binary
        if any(x for x in y.unique() if x not in [0, 1]):
            raise ValueError(
                "This encoder is only designed for binary classification, values of y "
                "can be only 0 or 1."
            )

        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        self.encoder_dict_ = {}

        total_pos = temp["target"].sum()
        total_neg = len(temp) - total_pos
        temp["non_target"] = np.where(temp["target"] == 1, 0, 1)

        for var in self.variables:
            pos = temp.groupby([var])["target"].sum() / total_pos
            neg = temp.groupby([var])["non_target"].sum() / total_neg

            t = pd.concat([pos, neg], axis=1)
            t["woe"] = np.log(t["target"] / t["non_target"])

            if (
                not t.loc[t["target"] == 0, :].empty
                or not t.loc[t["non_target"] == 0, :].empty
            ):
                raise ValueError(
                    "The proportion of one of the classes for a category in "
                    "variable {} is zero, and log of zero is not defined".format(var)
                )

            self.encoder_dict_[var] = t["woe"].to_dict()

        self._check_encoding_dictionary()

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().inverse_transform(X)

        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__
