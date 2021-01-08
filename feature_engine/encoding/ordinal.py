# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class OrdinalEncoder(BaseCategoricalTransformer):
    """
    The OrdinalCategoricalEncoder() replaces categories by ordinal numbers
    (0, 1, 2, 3, etc). The numbers can be ordered based on the mean of the target
    per category, or assigned arbitrarily.

    **Ordered ordinal encoding**: for the variable colour, if the mean of the target
    for blue, red and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 1,
    red by 2 and grey by 0.

    **Arbitrary ordinal encoding**: the numbers will be assigned arbitrarily to the
    categories, on a first seen first served basis.

    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed, the
    encoder will find and encode all categorical variables (type 'object').

    The encoder first maps the categories to the numbers for each variable (fit). The
    encoder then transforms the categories to the mapped numbers (transform).

    Parameters
    ----------
    encoding_method : str, default='ordered'
        Desired method of encoding.

        'ordered': the categories are numbered in ascending order according to
        the target mean value per category.

        'arbitrary' : categories are numbered arbitrarily.

    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and select all object type variables.

    Attributes
    ----------
    encoder_dict_ :
        Dictionary with the ordinal number per category, per variable.

    Methods
    -------
    fit:
        Find the integer to replace each category in each variable.
    transform:
        Encode the categories to numbers.
    fit_transform:
        Fit to the data, then transform it.
    inverse_transform:
        Encode the numbers into the original categories.

    Notes
    -----
    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder

    References
    ----------
    Encoding into integers ordered following target mean was discussed in the following
    talk at PyData London 2017:

    .. [1] Galli S. "Machine Learning in Financial Risk Assessment".
        https://www.youtube.com/watch?v=KHGGlozsRtA
    """

    def __init__(
        self,
        encoding_method: str = "ordered",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if encoding_method not in ["ordered", "arbitrary"]:
            raise ValueError(
                "encoding_method takes only values 'ordered' and 'arbitrary'"
            )

        self.encoding_method = encoding_method
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learn the numbers to be used to replace the categories in each
        variable.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to be encoded.

        y : pandas series, default=None
            The Target. Can be None if encoding_method = 'arbitrary'.
            Otherwise, y needs to be passed when fitting the transformer.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame.
            - If any user provided variable is not categorical
        ValueError
            - If there are no categorical variables in the df or the df is empty
            - If the variable(s) contain null values

        Returns
        -------
        self
        """

        X = self._check_fit_input_and_variables(X)

        # join target to predictor variables
        if self.encoding_method == "ordered":
            if y is None:
                raise ValueError("Please provide a target y for this encoding method")

            temp = pd.concat([X, y], axis=1)
            temp.columns = list(X.columns) + ["target"]

        # find mappings
        self.encoder_dict_ = {}

        for var in self.variables:

            if self.encoding_method == "ordered":
                t = (
                    temp.groupby([var])["target"]
                    .mean()
                    .sort_values(ascending=True)
                    .index
                )

            elif self.encoding_method == "arbitrary":
                t = X[var].unique()

            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

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
