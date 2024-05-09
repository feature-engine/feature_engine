import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from feature_engine.creation.distance_features import DistanceFeatures


@pytest.mark.parametrize(
    "input_data, expected_data, output_column_name, drop_original",
    [
        (
            {
                "a_latitude": [0.0, 0.0, 46.948579],
                "a_longitude": [0.0, 0.0, 7.436925],
                "b_latitude": [0.0, 12.34, 59.91054],
                "b_longitude": [0.0, 123.45, 10.752695],
            },
            {
                "distance_between_a_and_b": [0.0, 13630.28, 1457.49],
            },
            "distance_between_a_and_b",
            False,
        )
    ],
)
def test_compute_distance_without_dropping_lat_lon_columns(
    input_data,
    expected_data,
    output_column_name,
    drop_original,
):
    input_df = pd.DataFrame(input_data)
    expected_df = pd.DataFrame(input_data | expected_data)

    distance_transformer = DistanceFeatures(
        coordinate_columns=[["a_latitude", "a_longitude", "b_latitude", "b_longitude"]],
        output_column_names=[output_column_name],
        drop_original=drop_original,
    )

    distance_transformer.fit(input_df)
    output_df = distance_transformer.transform(X=input_df)

    assert_frame_equal(output_df, expected_df)


@pytest.mark.parametrize(
    "input_data, expected_data, output_column_name, drop_original",
    [
        (
            {
                "a_latitude": [0.0, 0.0, 46.948579],
                "a_longitude": [0.0, 0.0, 7.436925],
                "b_latitude": [0.0, 12.34, 59.91054],
                "b_longitude": [0.0, 123.45, 10.752695],
            },
            {
                "distance_between_a_and_b": [0.0, 13630.28, 1457.49],
            },
            "distance_between_a_and_b",
            True,
        )
    ],
)
def test_compute_distance_with_dropping_lat_lon_columns(
    input_data,
    expected_data,
    output_column_name,
    drop_original,
):
    input_df = pd.DataFrame(input_data)
    expected_df = pd.DataFrame(expected_data)

    distance_transformer = DistanceFeatures(
        coordinate_columns=[["a_latitude", "a_longitude", "b_latitude", "b_longitude"]],
        output_column_names=[output_column_name],
        drop_original=drop_original,
    )

    distance_transformer.fit(input_df)
    output_df = distance_transformer.transform(X=input_df)

    assert_frame_equal(output_df, expected_df)


@pytest.mark.parametrize(
    "input_data, expected_data, output_column_name, drop_original",
    [
        (
            {
                "a_latitude": [0.0, 0.0, 46.948579],
                "a_longitude": [0.0, 0.0, 7.436925],
                "b_latitude": [0.0, 12.34, 59.91054],
                "b_longitude": [0.0, 123.45, 10.752695],
                "c_latitude": [0.0, 0.0, 46.948579],
                "c_longitude": [0.0, 0.0, 7.436925],
                "d_latitude": [0.0, 12.34, 59.91054],
                "d_longitude": [0.0, 123.45, 10.752695],
            },
            {
                "distance_between_a_and_b": [0.0, 13630.28, 1457.49],
                "distance_between_c_and_d": [0.0, 13630.28, 1457.49],
            },
            ["distance_between_a_and_b", "distance_between_c_and_d"],
            False,
        )
    ],
)
def test_compute_distance_multiple_coordinates(
    input_data,
    expected_data,
    output_column_name,
    drop_original,
):
    input_df = pd.DataFrame(input_data)
    expected_df = pd.DataFrame(input_data | expected_data)

    distance_transformer = DistanceFeatures(
        coordinate_columns=[
            ["a_latitude", "a_longitude", "b_latitude", "b_longitude"],
            ["c_latitude", "c_longitude", "d_latitude", "d_longitude"],
        ],
        output_column_names=output_column_name,
        drop_original=drop_original,
    )

    distance_transformer.fit(input_df)
    output_df = distance_transformer.transform(X=input_df)

    assert_frame_equal(output_df, expected_df)


@pytest.mark.parametrize(
    "input_data, output_column_name, drop_original",
    [
        (
            {
                "a_latitude": [6, 7, 5],
                "a_longitude": [3, 7, 9],
                "b_latitude": [0, 0, 0],
                "b_longitude": [0, 0, 0],
            },
            "distance_between_a_and_b",
            True,
        )
    ],
)
def test_output_column_name(input_data, output_column_name, drop_original):
    input_df = pd.DataFrame(input_data)

    distance_transformer = DistanceFeatures(
        coordinate_columns=[["a_latitude", "a_longitude", "b_latitude", "b_longitude"]],
        output_column_names=[output_column_name],
        drop_original=drop_original,
    )

    distance_transformer.fit(input_df)
    output_df = distance_transformer.transform(X=input_df)

    error_msg = f"column_name: {output_column_name} is not in {output_df.columns}"
    assert output_column_name in output_df.columns, error_msg


@pytest.mark.parametrize(
    "input_data",
    [
        {
            "a_latitude": [0, -100.0],
            "a_longitude": [0, 0],
            "b_latitude": [0, 0],
            "b_longitude": [0, 0],
        },
    ],
)
def test_latitude_is_incorrect(input_data):
    input_df = pd.DataFrame(input_data)
    with pytest.raises(ValueError):
        transformer = DistanceFeatures(
            coordinate_columns=[
                ["a_latitude", "a_longitude", "b_latitude", "b_longitude"],
            ],
            output_column_names=["distance_between_a_and_b"],
            drop_original=False,
        )
        transformer.fit(input_df)
        transformer.transform(X=input_df)


@pytest.mark.parametrize(
    "input_data",
    [
        {
            "a_latitude": [0, 0],
            "a_longitude": [-1_000, 0],
            "b_latitude": [0, 0],
            "b_longitude": [0, 0],
        },
    ],
)
def test_longitude_is_incorrect(input_data):
    input_df = pd.DataFrame(input_data)
    with pytest.raises(ValueError):
        transformer = DistanceFeatures(
            coordinate_columns=[
                ["a_latitude", "a_longitude", "b_latitude", "b_longitude"],
            ],
            output_column_names=["distance_between_a_and_b"],
            drop_original=False,
        )
        transformer.fit(input_df)
        transformer.transform(X=input_df)


@pytest.mark.parametrize(
    "input_data",
    [
        {
            "a_latitude": [0, 0],
            "a_longitude": [None, 0],
            "b_latitude": [0, 0],
            "b_longitude": [0, 0],
        },
    ],
)
def test_fit_raises_error_if_na_in_df(input_data):
    input_df = pd.DataFrame(input_data)
    with pytest.raises(ValueError):
        transformer = DistanceFeatures(
            coordinate_columns=[
                ["a_latitude", "a_longitude", "b_latitude", "b_longitude"],
            ],
            output_column_names=["distance_between_a_and_b"],
            drop_original=False,
        )
        transformer.fit(input_df)
        transformer.transform(X=input_df)


@pytest.mark.parametrize(
    "input_data",
    [
        {
            "a_latitude": [0, 0],
            "a_longitude": [0, 0],
            "b_latitude": [0, 0],
            "b_longitude": [0, 0],
        },
    ],
)
def test_fit_raises_error_if_lat_lon_columns_not_in_df(input_data):
    input_df = pd.DataFrame(input_data)
    with pytest.raises(ValueError):
        transformer = DistanceFeatures(
            coordinate_columns=[
                ["a_latitude", "a_longitude", "d_latitude", "d_longitude"],
            ],
            output_column_names=["distance_between_a_and_b"],
            drop_original=False,
        )
        transformer.fit(input_df)
        transformer.transform(X=input_df)


def test_raises_error_for_coordinate_columns_empty_list():
    with pytest.raises(ValueError):
        DistanceFeatures(
            coordinate_columns=[],
            output_column_names=["distance_between_a_and_b"],
            drop_original=False,
        )


def test_raises_error_for_drop_original_not_boolean():
    with pytest.raises(ValueError):
        DistanceFeatures(
            coordinate_columns=[
                ["a_latitude", "a_longitude", "b_latitude", "b_longitude"],
            ],
            output_column_names=["distance_between_a_and_b"],
            drop_original=0,
        )


def test_raises_error_for_missing_columns_names():
    with pytest.raises(ValueError):
        DistanceFeatures(
            coordinate_columns=[
                ["a_latitude", "a_longitude", "b_latitude", "b_longitude"],
                ["c_latitude", "c_longitude", "d_latitude", "d_longitude"],
            ],
            output_column_names=["distance_between_a_and_b"],
            drop_original=False,
        )


def test_raises_error_for_wrong_column_names_type():
    with pytest.raises(ValueError):
        DistanceFeatures(
            coordinate_columns=[
                ["a_latitude", "a_longitude", "b_latitude", "b_longitude"],
            ],
            output_column_names="[123]",
            drop_original=False,
        )
