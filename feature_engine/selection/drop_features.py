from typing import List, Union
import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.selection.base_selector import BaseSelector


class DropFeatures(BaseSelector):
    """
    DropFeatures() drops a list of variable(s) indicated by the user from the dataframe.

    **When is this transformer useful?**

    Sometimes, we create new variables combining other variables in the dataset, for
    example, we obtain the variable `age` by subtracting `date_of_application` from
    `date_of_birth`. After we obtained our new variable, we do not need the date
    variables in the dataset any more. Thus, we can add DropFeatures() in the Pipeline
    to have these removed.

    Parameters
    ----------
    features_to_drop : str or list, default=None
        Variable(s) to be dropped from the dataframe

    Methods
    -------
    fit:
        This transformer does not learn any parameter.
    transform:
        Drops indicated features.
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(self, features_to_drop: List[Union[str, int]]):

        if not isinstance(features_to_drop, list) or len(features_to_drop) == 0:
            raise ValueError(
                "features_to_drop should be a list with the name of the variables"
                "you wish to drop from the dataframe."
            )

        self.features_to_drop = features_to_drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        This transformer does not learn any parameter.

        Verifies that the input X is a pandas dataframe, and that the variables to
        drop exist in the training dataframe.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input dataframe
        y : pandas Series, default = None
            y is not needed for this transformer. You can pass y or None.

        Returns
        -------
        self
        """
        # check input dataframe
        X = _is_dataframe(X)

        # X[self.features_to_drops] calls to pandas to check if columns are
        # present in the df.
        X[self.features_to_drop]

        self.features_to_drop_ = self.features_to_drop

        # check user is not removing all columns in the dataframe
        if len(self.features_to_drop) == len(X.columns):
            raise ValueError(
                "The resulting dataframe will have no columns after dropping all "
                "existing variables"
            )

        # add input shape
        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseSelector.transform.__doc__
