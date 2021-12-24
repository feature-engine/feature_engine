# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Dict, List, Optional, Union

import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.validation import _return_tags
from feature_engine.discretisation import BaseDiscretiser


class ArbitraryDiscretiser(BaseDiscretiser):
    """
    The ArbitraryDiscretiser() divides numerical variables into intervals which limits
    are determined by the user. Thus, it works only with numerical variables.

    You need to enter a dictionary with variable names as keys, and a list with
    the limits of the intervals as values. For example `{'var1':[0, 10, 100, 1000],
    'var2':[5, 10, 15, 20]}`.

    The ArbitraryDiscretiser() will then sort var1 values into the intervals 0-10,
    10-100, 100-1000, and var2 into 5-10, 10-15 and 15-20. Similar to `pandas.cut`.

    More details in the :ref:`User Guide <arbitrary_discretiser>`.

    Parameters
    ----------
    binning_dict: dict
        The dictionary with the variable to interval limits pairs. A valid dictionary
        looks like this:
        `binning_dict = {'var1':[0, 10, 100, 1000], 'var2':[5, 10, 15, 20]}`

    return_object: bool, default=False
        Whether the the discrete variable should be returned as numeric or as object.
        If you would like to proceed with the engineering of the variable as if
        it was categorical, use True. Alternatively, keep the default to False.

    return_boundaries: bool, default=False
        Whether the output, that is the bins, should be the interval boundaries. If
        True, it returns the interval boundaries. If False, it returns integers.

    Attributes
    ----------
    binner_dict_:
         Dictionary with the interval limits per variable.

    variables_:
         The variables that will be discretised.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        This transformer does not learn any parameter.
    transform:
        Sort variable values into the intervals.
    fit_transform:
        Fit to the data, then transform it.

    See Also
    --------
    pandas.cut
    """

    def __init__(
        self,
        binning_dict: Dict[Union[str, int], List[Union[str, int]]],
    ) -> None:

        if not isinstance(binning_dict, dict):
            raise ValueError(
                "Please provide at a dictionary with the interval limits per variable"
            )

        super().__init__(return_object, return_boundaries)

        self.binning_dict = binning_dict


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: None
            y is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = super()._select_variables_from_dict(X, self.binning_dict)

        # for consistency wit the rest of the discretisers, we add this attribute
        self.binner_dict_ = self.binning_dict

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sort the variable values into the intervals.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform variables
        if self.return_boundaries:
            for feature in self.variables_:
                X[feature] = pd.cut(X[feature], self.binner_dict_[feature])

        else:
            for feature in self.variables_:
                X[feature] = pd.cut(
                    X[feature], self.binner_dict_[feature], labels=False
                )

            # return object
            if self.return_object:
                X[self.variables_] = X[self.variables_].astype("O")

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        return tags_dict
