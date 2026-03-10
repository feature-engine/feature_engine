# Authors: Ankit Hemant Lade (contributor)
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine.dataframe_checks import (
    _check_optional_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


class GroupStandardScaler(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    GroupStandardScaler() scales numerical variables relative to a group (e.g.,
    by standardizing them to have a mean of 0 and a standard deviation of 1
    within each group).

    The transformer takes a list of numerical `variables` to standardise and a list
    of `reference` variables to group by. During fit, it learns the mean and
    standard deviation of each variable per group. During transform, it scales the
    variables applying the standard z-score formula per group.

    Unseen groups during `transform` will be scaled using the global mean and
    standard deviation learned during `fit`.

    More details in the :ref:`User Guide <group_standard_scaler>`.

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to be scaled. If None, the transformer
        will automatically find and select all numerical variables in the dataframe,
        except those specified in the `reference` parameter.

    reference: str or list
        The list of variables to use as the grouping key. These variables can be
        categorical or numerical.

    Attributes
    ----------
    barycenter_:
        Dictionary with the mean value per group for each variable.
        e.g. `{'var1': {grp1: 1.5, grp2: 3.0}}`
    scale_:
        Dictionary with the standard deviation per group for each variable.
        e.g. `{'var1': {grp1: 0.5, grp2: 1.0}}`

    global_mean_:
        Dictionary with the global mean value for each variable (for unseen groups).

    global_std_:
        Dictionary with the global standard deviation for each variable.

    variables_:
        The group of variables that will be transformed.

    reference_:
        The variables used to perform the grouping.

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find the mean and standard deviation per group for each variable.

    fit_transform:
        Fit to data, then transform it.

    transform:
        Standardise the variables relative to their group.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.scaling import GroupStandardScaler
    >>> X = pd.DataFrame(dict(
    ...     x1 = [1, 2, 3, 10, 20, 30],
    ...     grp = ['A', 'A', 'A', 'B', 'B', 'B']
    ... ))
    >>> gss = GroupStandardScaler(variables=['x1'], reference=['grp'])
    >>> gss.fit(X)
    >>> gss.transform(X)
        x1 grp
    0 -1.0   A
    1  0.0   A
    2  1.0   A
    3 -1.0   B
    4  0.0   B
    5  1.0   B
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        reference: Optional[Union[int, str, List[Union[str, int]]]] = None,
    ) -> None:

        if reference is None:
            raise ValueError("Parameter `reference` must be provided.")

        self.variables = _check_variables_input_value(variables)
        self.reference = _check_variables_input_value(reference)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the mean and standard deviation of each numerical variable per group.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The training input samples.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = check_X(X)

        self.reference_ = self.reference

        # Find variables to scale
        if self.variables is None:
            self.variables_ = find_numerical_variables(X)
            # Remove reference variables if they were automatically picked up
            self.variables_ = [
                var for var in self.variables_ if var not in self.reference_
            ]
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        # check that variables and reference are not overlapping
        overlapping = set(self.variables_).intersection(set(self.reference_))
        if overlapping:
            raise ValueError(
                f"Variables {overlapping} are specified in both `variables` and `reference`. "
                f"A variable cannot be both scaled and used as a grouping key."
            )

        # Check for missing values in variables and references
        _check_optional_contains_na(X, self.variables_ + self.reference_)

        # Calculate group means and standard deviations
        grouped = X.groupby(self.reference_)[self.variables_]

        self.barycenter_ = grouped.mean().to_dict()
        self.scale_ = grouped.std(ddof=1).to_dict()

        # Handle groups with only 1 element that cause std=NaN
        for var in self.variables_:
            for grp, val in self.scale_[var].items():
                if pd.isna(val):
                    self.scale_[var][grp] = 0.0
                elif val == 0:
                    self.scale_[var][grp] = 0.0  # Just making sure for consistency

        # Calculate global parameters for unseen groups
        self.global_mean_ = X[self.variables_].mean().to_dict()
        self.global_std_ = X[self.variables_].std(ddof=1).to_dict()

        for var in self.variables_:
            if pd.isna(self.global_std_[var]) or self.global_std_[var] == 0:
                self.global_std_[var] = 1.0

        # Save input features
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the variables relative to their group.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe
            The dataframe with the standardized variables.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check input data contains same number of columns as df used to fit
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        _check_optional_contains_na(X, self.variables_ + self.reference_)

        # reorder df to match train set
        X = X[self.feature_names_in_]

        X_transformed = X.copy()

        # We create a temporary grouping series if multiple references
        if len(self.reference_) == 1:
            group_keys = X_transformed[self.reference_[0]]
        else:
            group_keys = pd.Series(
                list(zip(*[X_transformed[col] for col in self.reference_])),
                index=X_transformed.index
            )

        for var in self.variables_:
            # Extract means and stds for the groups found in X
            means = group_keys.map(self.barycenter_[var])
            stds = group_keys.map(self.scale_[var])

            # Fill in global mean and std for groups not seen during `fit`
            means = means.fillna(self.global_mean_[var])
            stds = stds.fillna(self.global_std_[var])

            # Also replace 0 standard deviation with 1 to avoid division by zero
            stds = stds.replace(0, 1)

            # Standardise
            X_transformed[var] = (X_transformed[var] - means) / stds

        return X_transformed

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Get output feature names for transformation."""
        check_is_fitted(self)
        return list(self.feature_names_in_)

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        # This transformer has mandatory parameters (reference)
        tags_dict["_xfail_checks"]["check_parameters_default_constructible"] = (
            "transformer has mandatory parameters"
        )
        return tags_dict
