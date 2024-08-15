from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _binner_dict_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _variables_numerical_docstring,
)
from feature_engine._docstrings.init_parameters.discretisers import (
    _precision_docstring,
    _return_boundaries_docstring,
    _return_object_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_discretiser_docstring,
    _fit_transform_docstring,
    _transform_discretiser_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.discretisation.base_discretiser import BaseDiscretiser


@Substitution(
    return_object=_return_object_docstring,
    return_boundaries=_return_boundaries_docstring,
    precision=_precision_docstring,
    binner_dict_=_binner_dict_docstring,
    fit=_fit_discretiser_docstring,
    transform=_transform_discretiser_docstring,
    variables=_variables_numerical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class BinaryDiscretiser(BaseDiscretiser):
    """
    The BinaryDiscretiser() divides continuous numerical variables into two intervals,
    where the value `threshold`, the point at which the interval is  divided, is
    determined by the user.

    The BinaryDiscretiser() works only with numerical variables.
    A list of variables can be passed as argument. Alternatively, the discretiser
    will automatically select all numerical variables.

    The BinaryDiscretiser() first finds the boundaries for the intervals for
    each variable. Then, it transforms the variables, that is, sorts the values into
    the intervals.

    Parameters
    ----------
    {variables}

    threshold: int, float, default=None
        Desired value at which to divide the interval.

    {return_object}

    {return_boundaries}

    {precision}

    Attributes
    ----------
    {binner_dict_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    See Also
    --------
    pandas.cut
    sklearn.preprocessing.KBinsDiscretizer

    References
    ----------
    .. [1] Kotsiantis and Pintelas, "Data preprocessing for supervised leaning,"
        International Journal of Computer Science,  vol. 1, pp. 111 117, 2006.

    .. [2] Dong. "Beating Kaggle the easy way". Master Thesis.
        https://www.ke.tu-darmstadt.de/lehre/arbeiten/studien/2015/Dong_Ying.pdf

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.discretisation import EqualWidthDiscretiser
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.randint(1,100, 100)))
    >>> transformer = BinaryDiscretiser(threshold=50)
    >>> transformer.fit(X)
    >>> transformer.transform(X)['x'].value_counts()
        x
        1    56
        0    44
        Name: count, dtype: int64
    """

    def __init__(
        self,
        threshold: Union[None, int, float] = None,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_object: bool = False,
        return_boundaries: bool = False,
        precision: int = 3,
    ) -> None:

        if threshold is None:
            raise TypeError(
                "threshold not supplied."
                " Please provide a threshold of type float or int."
            )

        if not isinstance(threshold, (int, float)):
            raise TypeError(
                "threshold must be an integer or a float."
                f" Got type '{type(threshold).__name__}' instead."
            )

        super().__init__(return_object, return_boundaries, precision)

        self.variables = _check_variables_input_value(variables)
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the boundaries of the bins for each
        variable.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the variables
            to be transformed.
        y: None
            y is not needed in this encoder. You can pass y or None.
        """

        # check input dataframe
        X = super().fit(X)

        failed_threshold_check = []
        self.binner_dict_ = {}
        for var in self.variables_:
            # Check that threshold is within range
            if (self.threshold < min(X[var])) or (self.threshold > max(X[var])):
                # Omit these features from transformation step
                failed_threshold_check.append(var)
            else:
                self.binner_dict_[var] = [
                    float("-inf"),
                    np.float64(self.threshold),
                    float("inf"),
                ]

        if failed_threshold_check:
            print(
                "threshold outside of range for one or more variables."
                f" Features {failed_threshold_check} have not been transformed."
            )

        # A list of features that satisfy threshold check and will be transformed
        self.variables_trans_ = [
            var for var in self.variables_ if var not in failed_threshold_check
        ]

        return self

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
            for feature in self.variables_trans_:
                X[feature] = pd.cut(
                    X[feature],
                    self.binner_dict_[feature],
                    precision=self.precision,
                    include_lowest=True,
                )
            X[self.variables_trans_] = X[self.variables_trans_].astype(str)

        else:
            for feature in self.variables_trans_:
                X[feature] = pd.cut(
                    X[feature],
                    self.binner_dict_[feature],
                    labels=False,
                    include_lowest=True,
                )

            # return object
            if self.return_object:
                X[self.variables_trans_] = X[self.variables_trans_].astype("O")

        return X
