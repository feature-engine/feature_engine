# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

import pandas as pd

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer


class BaseDiscretiser(BaseNumericalTransformer):
    """
    Shared set-up checks and methods across numerical discretisers.

    Important: inherits fit() functionality and tags from BaseNumericalTransformer.
    """

    def __init__(
        self,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
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

        if not isinstance(precision, int) or precision < 1:
            raise ValueError(
                "precision must be a positive integer. " f"Got {precision} instead."
            )

        self.return_object = return_object
        self.return_boundaries = return_boundaries
        self.precision = precision

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
        X = self._check_transform_input_and_state(X)

        # transform variables
        if self.return_boundaries is True:
            for feature in self.variables_:
                X[feature] = pd.cut(
                    X[feature],
                    self.binner_dict_[feature],
                    precision=self.precision,
                    include_lowest=True,
                )
            X[self.variables_] = X[self.variables_].astype(str)

        else:
            for feature in self.variables_:
                X[feature] = pd.cut(
                    X[feature],
                    self.binner_dict_[feature],
                    labels=False,
                    include_lowest=True,
                )

            # return object
            if self.return_object:
                X[self.variables_] = X[self.variables_].astype("O")

        return X
