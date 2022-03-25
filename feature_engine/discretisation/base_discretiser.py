# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer


class BaseDiscretiser(BaseNumericalTransformer):
    """
    Shared set-up checks and methods across numerical discretisers.

    Important: inherits fit() functionality and tags from BaseNumericalTransformer.
    """

    _return_object_docstring = """return_object: bool, default=False
        Whether the the discrete variable should be returned as numeric or as
        object. If you would like to proceed with the engineering of the variable as if
        it was categorical, use True. Alternatively, keep the default to False.
        """.rstrip()

    _return_boundaries_docstring = """return_boundaries: bool, default=False
        Whether the output should be the interval boundaries. If True, it returns
        the interval boundaries. If False, it returns integers.
        """.rstrip()

    _binner_dict_docstring = """binner_dict_:
         Dictionary with the interval limits per variable.
         """.rstrip()

    _fit_docstring = """fit:
        Find the interval limits.
        """.rstrip()

    _transform_docstring = """transform:
        Sort continuous variable values into the intervals.
    """

    def __init__(
        self,
        return_object: bool = False,
        return_boundaries: bool = False,
    ) -> None:

        if not isinstance(return_object, bool):
            raise ValueError(
                "return_object must be True or False. " f"Got {return_object} instead."
            )

        if not isinstance(return_boundaries, bool):
            raise ValueError(
                "return_boundaries must be True or False. "
                f"Got {return_boundaries} instead."
            )

        self.return_object = return_object
        self.return_boundaries = return_boundaries

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sort the variable values into the intervals.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform variables
        if self.return_boundaries is True:
            for feature in self.variables_:
                X[feature] = pd.cut(X[feature], self.binner_dict_[feature])
            X[self.variables_] = X[self.variables_].astype(str)

        else:
            for feature in self.variables_:
                X[feature] = pd.cut(
                    X[feature], self.binner_dict_[feature], labels=False
                )

            # return object
            if self.return_object:
                X[self.variables_] = X[self.variables_].astype("O")

        return X
