from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._base_transformers.mixins import (
    FitFromDictMixin,
    GetFeatureNamesOutMixin,
)


class DistanceFeatures(
    BaseNumericalTransformer, FitFromDictMixin, GetFeatureNamesOutMixin
):
    """
    DistanceFeatures() computes the distance between pairs of columns containing
    coordinates. The distance between two pairs of coordinates is computed using the
    Haversine formula (or the great circle formula).

    The Haversine formula is not the most precise way to compute the distance between
    two points on the Earth. However, it is precise enough for our purposes and is fast.

    DistanceFeatures() requires a list of column names of coordinates, i.e., a list of
    lists of 4 elements, where each 4-list represents the column names of the pair of
    coordinates for which we should compute the distance. Additionally, it is possible
    to provide the names of the columns contaning the distances and chose if the
    coordinate columns are dropped or not.

    Missing data should be imputed before using this transformer.



    Parameters
    ----------
    coordinate_columns: List[List[Union[str, int]]],

    output_column_names: List[Union[str, None]], default=None
        List of names for the column with the computed distance. Note that the list
        must have equal length to the `coordinate_columns` list. This is because the
        transformer need to know which distance column has which name. If none, the
        default names are generated.

    drop_original: Optional[bool], default=False
        If True, then the `coordinate_columns` columns are dropped. Otherwise,
        they are left in the dataframe.

    Attributes
    ----------
    ...

    Methods
    -------
    fit:
        Learns the variable's maximum values.

    transform:
        Compute the distance using the coordinates provided in the `coordinate_columns`.

    References
    ----------
    https://en.wikipedia.org/wiki/Haversine_formula

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.creation import DistanceFeatures
    >>> X = pd.DataFrame({
            'a_latitude': [0., 0., 46.948579],
            'a_longitude': [0., 0., 7.436925],
            'b_latitude': [0., 12.34, 59.91054],
            'b_longitude': [0., 123.45, 10.752695],
        })
    >>> cf = DistanceFeatures(
        coordinate_columns=[['a_latitude', 'a_longitude', 'b_latitude', 'b_longitude']],
        output_column_names=['distance_between_a_and_b'],
        drop_original=False,
    )
    >>> cf.fit(X)
    >>> cf.transform(X)
       a_latitude   a_longitude   b_latitude   b_longitude     distance_between_a_and_b
    0       0.           0.           0.           0.                   0.
    1       0.           0.          12.34       123.45             13630.28
    2      46.94         7.43        59.91        10.75              1457.49
    """

    _EARTH_RADIUS: float = 6371.0  # radius of Earth in kms

    def __init__(
        self,
        coordinate_columns: List[List[Union[str, int]]],
        output_column_names: List[Union[str, None]] = None,
        drop_original: Optional[bool] = False,
    ) -> None:

        # the coordinate_columns variable is rewritten in this way to speed up
        # computation later, i.e., to use vectorization
        (self.a_latitudes, self.a_longitudes, self.b_latitudes, self.b_longitudes) = (
            self._check_coordinate_columns(columns=coordinate_columns)
        )
        self.output_column_name = self._check_output_columns_names(
            column_name=output_column_names,
            coordinate_columns=coordinate_columns,
        )

        self.drop_original = drop_original

        self.variables = None

    def _check_drop_original(self, parameter: bool) -> bool:
        if isinstance(parameter, bool) is False:
            raise ValueError(
                "Expected boolean value for parameter `drop_original`, "
                f"but got {parameter} with type {type(parameter)}"
            )

    def _check_coordinate_columns(
        self, columns: List[List[Union[str, int]]]
    ) -> Tuple[List[Union[str, int]], ...]:
        if not columns:
            raise ValueError("Empty list for `coordinate_columns` not allowed!")
        for idx, coordinate_column in enumerate(columns):
            if len(coordinate_column) != 4:
                raise ValueError(
                    f"Needed 4 values to compute a distance, "
                    f"but got {len(coordinate_column)} columns \n"
                    f"at the index {idx} of the list coordinate columns."
                )
        return (
            [coordinate[0] for coordinate in columns],
            [coordinate[1] for coordinate in columns],
            [coordinate[2] for coordinate in columns],
            [coordinate[3] for coordinate in columns],
        )

    def _check_output_columns_names(
        self,
        column_name: List[Union[str, int]],
        coordinate_columns: List[List[Union[str, int]]],
    ) -> List[Union[str, int]]:
        if column_name is None:
            return [f"distance_{c[0]}_{c[1]}_{c[2]}_{c[3]}" for c in coordinate_columns]
        if len(column_name) != len(coordinate_columns):
            raise ValueError(
                "Not enough output column names provided.\n "
                f"Expected {len(coordinate_columns)} column names, "
                f"but got {len(column_name)}."
            )
        return column_name

    def fit(self, X: pd.DataFrame):
        # there is no fit for this transformer
        super().fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        """
        Compute the distance on heart using the Haversine formula.

        Parameters
        ----------
        X: Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: Pandas dataframe.
            The original dataframe plus the distances between the given coordinates.
        """
        X = self._check_transform_input_and_state(X)
        X = self._check_lat_lon_columns_are_in_df(X)
        X = self._check_correctness_of_coordinates(X)

        self._compute_distance(X)

        if self.drop_original:
            X.drop(
                columns=[
                    *self.a_latitudes,
                    *self.a_longitudes,
                    *self.b_latitudes,
                    *self.b_longitudes,
                ],
                inplace=True,
            )

        return X

    def _check_lat_lon_columns_are_in_df(self, X) -> pd.DataFrame:
        coordinate_columns = [
            *self.a_latitudes,
            *self.a_longitudes,
            *self.b_latitudes,
            *self.b_longitudes,
        ]
        if set(coordinate_columns).issubset(set(X.columns)) is False:
            raise ValueError(
                f"The columns {set(coordinate_columns).issubset(set(X.columns))} "
                f"were not found in the dataframe."
            )
        return X

    def _check_correctness_of_coordinates(self, X: pd.DataFrame) -> pd.DataFrame:
        # recall that the latitude is a number between -90 and +90,
        # while longitudes is between -180 and +180.
        irregular_latitudes = (
            (X[[*self.a_latitudes, *self.b_latitudes]].abs() > 90).sum().sum()
        )
        irregular_longitudes = (
            (X[[*self.a_longitudes, *self.b_longitudes]].abs() > 180).sum().sum()
        )

        if irregular_latitudes > 0:
            raise ValueError("The dataframe contains irregular latitudes")
        elif irregular_longitudes > 0:
            raise ValueError("The dataframe contains irregular longitudes")
        else:
            return X

    def _compute_distance(self, X: pd.DataFrame):

        # convert latitude and longitude in radians
        phi_1 = np.radians(X[self.a_latitudes].to_numpy())
        phi_2 = np.radians(X[self.b_latitudes].to_numpy())
        lambda_1 = np.radians(X[self.a_longitudes].to_numpy())
        lambda_2 = np.radians(X[self.b_longitudes].to_numpy())

        # compute delta, i.e., difference, between radians
        delta_phi = phi_2 - phi_1
        delta_lambda = lambda_2 - lambda_1

        # compute distance using Haversine formula
        inner_part = (
            np.sin(delta_phi / 2) ** 2
            + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2) ** 2
        )
        X[self.output_column_name] = (
            self._EARTH_RADIUS * 2 * np.arcsin(np.sqrt(inner_part))
        )
