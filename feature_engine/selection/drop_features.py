from typing import List, Union

import pandas as pd

from feature_engine.dataframe_checks import check_X
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import check_all_variables


class DropFeatures(BaseSelector):
    """
    DropFeatures() drops a list of variables indicated by the user from the dataframe.

    More details in the :ref:`User Guide <drop_features>`.

    Parameters
    ----------
    features_to_drop: str or list
        Variable(s) to be dropped from the dataframe

    Attributes
    ----------
    features_to_drop_:
        The features that will be dropped.

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        This transformer does not learn any parameter.

    fit_transform:
        Fit to data, then transform it.

    get_feature_names_out:
        Get output feature names for transformation.

    get_support:
        Get a mask, or integer index, of the features selected.

    get_params:
        Get parameters for this estimator.

    set_params:
        Set the parameters of this estimator.

    transform:
        Drops indicated features.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.selection import DropFeatures
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4],
    >>>                         x2 = ["a", "a", "b", "c"],
    >>>                         x3 = [True, False, False, True]))
    >>> df = DropFeatures(features_to_drop=["x2"])
    >>> df.fit_transform(X)
        x1     x3
    0   1   True
    1   2  False
    2   3  False
    3   4   True
    """

    def __init__(self, features_to_drop: List[Union[str, int]]):
        if not isinstance(features_to_drop, (str, list)) or len(features_to_drop) == 0:
            raise ValueError(
                f"features_to_drop should be a list with the name of the variables "
                f"you wish to drop from the dataframe. Got {features_to_drop} instead."
            )

        self.features_to_drop = features_to_drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input dataframe
        y : pandas Series, default = None
            y is not needed for this transformer. You can pass y or None.
        """
        # check input dataframe
        X = check_X(X)

        self.features_to_drop_ = check_all_variables(X, variables=self.features_to_drop)

        # check user is not removing all columns in the dataframe
        if len(self.features_to_drop_) == len(X.columns):
            raise ValueError(
                "The resulting dataframe will have no columns after dropping all "
                "existing variables"
            )

        # save input features
        self._get_feature_names_in(X)

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"][
            "check_fit2d_1feature"
        ] = "the transformer raises an error when removing the only column, ok to fail"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
