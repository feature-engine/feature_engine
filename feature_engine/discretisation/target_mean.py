from typing import List, Union

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from feature_engine._docstrings.class_inputs import _variables_numerical_docstring
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
    check_X_y,
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
)
from feature_engine.discretisation.base_discretiser import BaseDiscretiser
from feature_engine.encoding import MeanEncoder
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)


@Substitution(
    return_objects=BaseDiscretiser._return_object_docstring,
    return_boundaries=BaseDiscretiser._return_boundaries_docstring,
    binner_dict_=BaseDiscretiser._binner_dict_docstring,
    transform=BaseDiscretiser._transform_docstring,
    variables=_variables_numerical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class MeanDiscretiser(BaseDiscretiser):
    """
    The MeanDiscretiser sorts numerical variables into intervals of equal-widht or
    equal-frequency, and then replaces the interval values by the mean target value
    within each interval.

    The method is inspired by the following article from one of the top solutions of
    the KDD 2009 competition:
    Predicting customer behaviour: The University of Melbourne’s KDD Cup report
    H. Miller, S. Clarke, S. Lane, A. Lonie, D. Lazaridis, S. Petrovski, O. Jones;
    JMLR W&CP 7:45–55, 2009.

    More details in the :ref:`User Guide <mean_discretiser>`.

    Parameters
    ----------
    {variables}

    strategy: str, default='equal_width'
        Whether the bins should be of equal width ('equal_width') or equal frequency
        ('equal_frequency').

    bins: int, default=10
        Desired number of equal-width or equal-distance intervals / bins.

    Attributes
    ----------
    {variables_}

    {binner_dict_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    See Also
    --------
    feature_engine.encoding.MeanEncoder

    References
    ----------
    .. [1] H. Miller, S. Clarke, S. Lane, A. Lonie, D. Lazaridis, S. Petrovski, O.
    Jones, "Predicting customer behaviour: The University of Melbourne’s KDD Cup
    report," JMLR W&CP 7:45–55, 2009.
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        bins: int = 10,
        strategy: str = "equal_frequency",
    ) -> None:

        if not isinstance(bins, int):
            raise ValueError(
                f"bins must be an integer. Got {bins} instead."
            )
        if strategy not in ("equal_frequency", "equal_width"):
            raise ValueError(
                "strategy must equal 'equal_frequency' or 'equal_width'. "
                f"Got {strategy} instead."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.bins = bins
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the interval boundaries and the mean value of the target for each
        interval.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y : pandas series of shape = [n_samples,]
            y is not needed in this discretiser. You can pass y or None.
        """
        # check if 'X' is a dataframe
        X, y = check_X_y(X, y)

        #  identify numerical variables
        self.variables_ = _find_or_check_numerical_variables(
            X, self.variables
        )

        # check for missing values
        _check_contains_na(X, self.variables_)

        # check for inf
        _check_contains_inf(X, self.variables_)

        # instantiate pipeline
        self._pipeline = self._make_pipeline()
        self._pipeline.fit(X, y)

        # TODO: to help the user understand the output of the transformation, we need
        # to create a dictionary that contains the variable as key, and as value a
        # tuple where the first item is the limits of the interval, and the second
        # item is the mean target of the value. We can obtain the interval limits from
        # the discretiser and the mean target values from the encoder.
        self.binner_dict_ = {}

        # store input features
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace original values by the average of the target mean value per bin
        for each of the variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_enc: pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the means of the selected numerical variables.

        """
        # check that fit method has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # check that input data contain number of columns as the fitted df
        _check_X_matches_training_df(X, self.n_features_in_)

        # check for missing values
        _check_contains_na(X, self.variables_)

        # check for infinite values
        _check_contains_inf(X, self.variables_)

        # discretise and encode
        X_tr = self._pipeline.transform(X)

        return X_tr

    def _make_discretiser(self):
        """
        Instantiate the EqualFrequencyDiscretiser or EqualWidthDiscretiser.
        """
        if self.strategy == "equal_frequency":
            discretiser = EqualFrequencyDiscretiser(
                q=self.bins,
                variables=self.variables_,
                return_boundaries=True,
                return_object=True,
            )
        else:
            discretiser = EqualWidthDiscretiser(
                bins=self.bins,
                variables=self.variables_,
                return_boundaries=True,
                return_object=True,
            )

        return discretiser

    def _make_pipeline(self):
        """
        Instantiate pipeline comprised of discretiser and encoder.
        """
        pipe = Pipeline([
            ("discretiser", self._make_discretiser()),
            ("encoder", MeanEncoder(variables=self.variables_))
        ])

        return pipe
