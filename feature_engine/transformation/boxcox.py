# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd
import scipy.stats as stats

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import _check_input_parameter_variables


class BoxCoxTransformer(BaseNumericalTransformer):
    """
    The BoxCoxTransformer() applies the BoxCox transformation to numerical
    variables.

    The Box-Cox transformation is defined as:

    - T(Y)=(Y exp(λ)−1)/λ if λ!=0
    - log(Y) otherwise

    where Y is the response variable and λ is the transformation parameter. λ varies,
    typically from -5 to 5. In the transformation, all values of λ are considered and
    the optimal value for a given variable is selected.

    The BoxCox transformation implemented by this transformer is that of
    SciPy.stats:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html

    The BoxCoxTransformer() works only with numerical positive variables (>=0).

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to transform. If None, the transformer will
        automatically find and select all numerical variables.

    Attributes
    ----------
    lambda_dict_:
        Dictionary with the best BoxCox exponent per variable.

    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Learn the optimal lambda for the BoxCox transformation.
    transform:
        Apply the BoxCox transformation.
    fit_transform:
        Fit to data, then transform it.

    References
    ----------
    .. [1] Box and Cox. "An Analysis of Transformations". Read at a RESEARCH MEETING,
        1964.
        https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1964.tb00553.x
    """

    def __init__(
        self, variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:

        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the optimal lambda for the BoxCox transformation.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
         TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            - If there are no numerical variables in the df or the df is empty
            - If the variable(s) contain null values
            - If some variables contain zero values

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)

        self.lambda_dict_ = {}

        for var in self.variables_:
            _, self.lambda_dict_[var] = stats.boxcox(X[var])

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the BoxCox transformation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If the df has different number of features than the df used in fit()
            - If some variables contain negative values

        Returns
        -------
        X: pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform
        for feature in self.variables_:
            X[feature] = stats.boxcox(X[feature], lmbda=self.lambda_dict_[feature])

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # =======  this tests fail because the transformers throw an error
        # when the values are 0. Nothing to do with the test itself but
        # mostly with the data created and used in the test
        msg = (
            "transformers raise errors when data contains zeroes, thus this check fails"
        )
        tags_dict["_xfail_checks"]["check_estimators_dtypes"] = msg
        tags_dict["_xfail_checks"]["check_estimators_fit_returns_self"] = msg
        tags_dict["_xfail_checks"]["check_pipeline_consistency"] = msg
        tags_dict["_xfail_checks"]["check_estimators_overwrite_params"] = msg
        tags_dict["_xfail_checks"]["check_estimators_pickle"] = msg
        tags_dict["_xfail_checks"]["check_transformer_general"] = msg

        # boxcox fails this test as well
        msg = "scipy.stats.boxcox does not like the input data"
        tags_dict["_xfail_checks"]["check_methods_subset_invariance"] = msg
        tags_dict["_xfail_checks"]["check_fit2d_1sample"] = msg

        return tags_dict
