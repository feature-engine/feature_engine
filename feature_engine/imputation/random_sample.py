# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import hashlib
from typing import List, Optional, Union

import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _transform_imputers_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import check_all_variables, find_all_variables


def _hash_seeds(values) -> np.ndarray:
    """Return one seed per row, in [0, 2**32), derived from the row's values.

    Rows with the same values get the same seed, regardless of their position.
    Values are compared as floats (25 and 25.0 are equal) and missing values
    count as 0. hashlib, unlike hash(), gives the same seed in every session.
    """
    values = np.asarray(values, dtype="float64")
    values = values.reshape(len(values), -1)
    # + 0.0 turns -0.0 into 0.0, so both give the same bytes
    values = np.where(np.isnan(values), 0.0, values) + 0.0
    values = np.ascontiguousarray(values, dtype="<f8")
    return np.array(
        [
            int.from_bytes(
                hashlib.blake2b(row.tobytes(), digest_size=4).digest(), "little"
            )
            for row in values
        ],
        dtype=np.int64,
    )


@Substitution(
    variables_=_variables_attribute_docstring,
    return_empty=_return_empty_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    transform=_transform_imputers_docstring,
    fit_transform=_fit_transform_docstring,
)
class RandomSampleImputer(BaseImputer):
    """
    The RandomSampleImputer() replaces missing data with a random sample extracted from
    the variables in the training set.

    The RandomSampleImputer() works with both numerical and categorical variables.

    **Note**

    The random samples used to replace missing values may vary from execution to
    execution. This may affect the results of your work. Thus, it is advisable to set a
    seed.

    More details in the :ref:`User Guide <random_sample_imputer>`.

    Parameters
    ----------
    variables: list, default=None
        The list of variables to be imputed. If None, the imputer will select
        all variables in the train set.

    {return_empty}

    random_state: int, str or list, default=None
        The random_state can take an integer to set the seed when extracting the
        random samples. Alternatively, it can take a variable name or a list of
        variables, whose values will be used to determine the seed, observation per
        observation.

    seed: str, default='general'
        Indicates whether the seed should be set for each observation with missing
        values, or if one seed should be used to impute all observations in one go.

        **'general'**: one seed will be used to impute the entire dataframe. This is
        equivalent to setting the seed in pandas.sample(random_state).

        **'observation'**: the seed will be set for each observation from the values
        of the variables indicated in the random_state for that particular
        observation. Observations with the same values in those variables receive
        the same imputation, regardless of their position in the dataframe.
        Missing values in those variables are treated as 0.

    Attributes
    ----------
    X_:
        Copy of the training dataframe from which to extract the random samples.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Make a copy of the train set.

    {fit_transform}

    {transform}

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.imputation import RandomSampleImputer
    >>> X = pd.DataFrame(dict(
    >>>        x1 = [np.nan,1,1,0,np.nan],
    >>>        x2 = ["a", np.nan, "b", np.nan, "a"],
    >>>        ))
    >>> rsi = RandomSampleImputer(random_state=42)
    >>> rsi.fit(X)
    >>> rsi.transform(X)
        x1 x2
    0  0.0  a
    1  1.0  a
    2  1.0  b
    3  0.0  a
    4  1.0  a

    With polars:

    >>> import polars as pl
    >>> X = pl.DataFrame(dict(
    ...        x1 = [None, 1, 1, 0, None],
    ...        x2 = ["a", None, "b", None, "a"],
    ...        ))
    >>> rsi = RandomSampleImputer(random_state=42)
    >>> rsi.fit(X)
    >>> rsi.transform(X)
    shape: (5, 2)
    ┌─────┬─────┐
    │ x1  ┆ x2  │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 0   ┆ a   │
    │ 1   ┆ a   │
    │ 1   ┆ b   │
    │ 0   ┆ a   │
    │ 1   ┆ a   │
    └─────┴─────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        random_state: Union[None, int, str, List[Union[str, int]]] = None,
        seed: str = "general",
    ) -> None:

        if not isinstance(seed, str) or seed not in ["general", "observation"]:
            raise ValueError(
                "seed takes only values 'general' or 'observation'. "
                f"Got {seed} instead."
            )

        if seed == "general" and random_state:
            if not isinstance(random_state, int):
                raise ValueError(
                    "if seed == 'general' then random_state must take an integer. "
                    f"Got {random_state} instead."
                )

        if seed == "observation" and not random_state:
            raise ValueError(
                "if seed == 'observation' the random state must take the name of one "
                "or more variables which will be used to seed the imputer. "
                f"Got {random_state} instead."
            )

        self.variables = _check_variables_input_value(variables)

        _check_return_empty_is_bool(return_empty)
        self.return_empty = return_empty

        self.random_state = random_state
        self.seed = seed

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Makes a copy of the train set. Only stores a copy of the variables to impute.
        This copy is then used to randomly extract the values to fill the missing data
        during transform.

        Parameters
        ----------

        X: dataframe of shape = [n_samples, n_features]
            The training dataset. Can be a pandas, polars, or any other dataframe
            supported by narwhals.

        y: None
            y is not needed in this imputation. You can pass None or y.
        """

        # check input dataframe
        nw_X = check_X(X)

        # find variables to impute
        if self.variables is None:
            variables_ = find_all_variables(X, self.return_empty)
        else:
            variables_ = check_all_variables(X, self.variables)

        # take a copy of the selected variables
        if nwd.is_pandas_dataframe(X):
            X_ = X[variables_].copy()
        else:
            X_ = nw_X.select(variables_)

        # check the variables assigned to the random state
        if self.seed == "observation":
            random_state = _check_variables_input_value(self.random_state)
            if isinstance(random_state, (int, str)):
                random_state = [random_state]
            if random_state and any(
                var for var in random_state if var not in X.columns
            ):
                raise ValueError(
                    "There are variables assigned as random state which are not part "
                    f"of the training dataframe. Got {self.random_state} instead."
                )
            self.random_state = random_state

        self.variables_ = variables_
        self.X_ = X_
        self._get_feature_names_in(X)

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Replace missing data with random values taken from the train set.

        Parameters
        ----------

        X: dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the transformed variables.
        """

        nw_X = self._transform(X)

        if nwd.is_pandas_dataframe(X):
            X = self._transform_pandas(X)
        else:
            X = self._transform_narwhals(nw_X)

        return X

    def _transform_pandas(self, X):
        # copy first: the .loc assignments below fill NaNs in place, and
        # BaseImputer._transform no longer returns a copy (#1002), so without
        # this the caller's dataframe (and self.X_ when it is the same object)
        # would be mutated.
        X = X.copy()

        # random sampling with a general seed
        if self.seed == "general":
            for feature in self.variables_:
                if X[feature].isnull().sum() > 0:
                    # determine number of data points to extract at random
                    n_samples = X[feature].isnull().sum()

                    # extract values
                    random_sample = (
                        self.X_[feature]
                        .dropna()
                        .sample(n_samples, replace=True, random_state=self.random_state)
                    )
                    # re-index: pandas needs this to add the values to the right
                    # observations
                    random_sample.index = X[X[feature].isnull()].index

                    # replace na
                    X.loc[X[feature].isnull(), feature] = random_sample

        # random sampling observation per observation
        elif self.seed == "observation" and self.random_state:
            # seeds come from the values before any variable is imputed; rows are
            # addressed by position, so duplicated index labels don't matter
            seeds = _hash_seeds(X[self.random_state].to_numpy())
            for feature in self.variables_:
                is_null = X[feature].isnull().to_numpy()
                if is_null.any():
                    pool = self.X_[feature].dropna()
                    positions = np.flatnonzero(is_null)
                    random_values = [
                        pool.sample(
                            1, replace=True, random_state=int(seeds[pos])
                        ).iloc[0]
                        for pos in positions
                    ]
                    X.iloc[positions, X.columns.get_loc(feature)] = random_values
        return X

    def _transform_narwhals(self, X):

        if self.seed == "general":
            for feature in self.variables_:
                col = X[feature]
                null_mask = col.is_null()
                n_samples = int(null_mask.sum())
                if n_samples > 0:
                    positions = null_mask.arg_true()
                    random_sample = (
                        self.X_[feature]
                        .drop_nulls()
                        .sample(
                            n_samples, with_replacement=True, seed=self.random_state
                        )
                    )
                    # reassign X so each variable's imputation is carried over
                    # to the next iteration
                    X = X.with_columns(col.scatter(positions, random_sample))

        elif self.seed == "observation" and self.random_state:
            # seeds come from the values before any variable is imputed
            internal_seeds = _hash_seeds(X.select(self.random_state).to_numpy())

            for feature in self.variables_:
                col = X[feature]
                null_mask = col.is_null()
                if int(null_mask.sum()) > 0:
                    positions = null_mask.arg_true().to_list()
                    pool = self.X_[feature].drop_nulls()
                    random_values = [
                        pool.sample(
                            1, with_replacement=True, seed=int(internal_seeds[pos])
                        ).item()
                        for pos in positions
                    ]
                    # reassign X so each variable's imputation is carried over
                    # to the next iteration
                    X = X.with_columns(col.scatter(positions, random_values))

        return X.to_native()

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
