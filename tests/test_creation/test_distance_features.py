import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from feature_engine.creation.distance_features import DistanceFeatures


@pytest.mark.parametrize(
    'input_data, expected_data, output_column_name, drop_original',
    [(
            {
                'a_latitude': [0., 0., 46.948579],
                'a_longitude': [0., 0., 7.436925],
                'b_latitude': [0., 12.34, 59.91054],
                'b_longitude': [0., 123.45, 10.752695],
            },
            {
                'distance_between_a_and_b': [0., 13630.28, 1457.49],
            },
            'distance_between_a_and_b',
            False,
    )]
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
        a_latitude='a_latitude',
        a_longitude='a_longitude',
        b_latitude='b_latitude',
        b_longitude='b_longitude',
        output_column_name=output_column_name,
        drop_original=drop_original,
    )

    distance_transformer.fit(input_df)
    output_df = distance_transformer.transform(X=input_df)

    assert_frame_equal(output_df, expected_df)


@pytest.mark.parametrize(
    'input_data, expected_data, output_column_name, drop_original',
    [(
            {
                'a_latitude': [0., 0., 46.948579],
                'a_longitude': [0., 0., 7.436925],
                'b_latitude': [0., 12.34, 59.91054],
                'b_longitude': [0., 123.45, 10.752695],
            },
            {
                'distance_between_a_and_b': [0., 13630.28, 1457.49],
            },
            'distance_between_a_and_b',
            True,
    )]
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
        a_latitude='a_latitude',
        a_longitude='a_longitude',
        b_latitude='b_latitude',
        b_longitude='b_longitude',
        output_column_name=output_column_name,
        drop_original=drop_original,
    )

    distance_transformer.fit(input_df)
    output_df = distance_transformer.transform(X=input_df)

    assert_frame_equal(output_df, expected_df)


@pytest.mark.parametrize(
    'input_data, output_column_name, drop_original',
    [(
            {
                'a_latitude': [6, 7, 5],
                'a_longitude': [3, 7, 9],
                'b_latitude': [0, 0, 0],
                'b_longitude': [0, 0, 0],
            },
            'distance_between_a_and_b',
            True,
    )]
)
def test_output_column_name(input_data, output_column_name, drop_original):
    input_df = pd.DataFrame(input_data)

    distance_transformer = DistanceFeatures(
        a_latitude='a_latitude',
        a_longitude='a_longitude',
        b_latitude='b_latitude',
        b_longitude='b_longitude',
        output_column_name=output_column_name,
        drop_original=drop_original,
    )

    distance_transformer.fit(input_df)
    output_df = distance_transformer.transform(X=input_df)

    assert output_column_name in output_df.columns, f'column_name: {output_column_name} ' \
                                                    f'is not in {output_df.columns} '


@pytest.mark.parametrize(
    'input_data',
    [
        {
            'a_latitude': [0, -100.],
            'a_longitude': [0, 0],
            'b_latitude': [0, 0],
            'b_longitude': [0, 0],
        },
    ]
)
def test_latitude_is_incorrect(input_data):
    input_df = pd.DataFrame(input_data)
    with pytest.raises(ValueError):
        transformer = DistanceFeatures(
            a_latitude='a_latitude',
            a_longitude='a_longitude',
            b_latitude='b_latitude',
            b_longitude='b_longitude',
            output_column_name='distance_between_a_and_b',
            drop_original=False,
        )
        transformer.fit(input_df)
        transformer.transform(X=input_df)


@pytest.mark.parametrize(
    'input_data',
    [
        {
            'a_latitude': [0, 0],
            'a_longitude': [-1_000, 0],
            'b_latitude': [0, 0],
            'b_longitude': [0, 0],
        },
    ]
)
def test_longitude_is_incorrect(input_data):
    input_df = pd.DataFrame(input_data)
    with pytest.raises(ValueError):
        transformer = DistanceFeatures(
            a_latitude='a_latitude',
            a_longitude='a_longitude',
            b_latitude='b_latitude',
            b_longitude='b_longitude',
            output_column_name='distance_between_a_and_b',
            drop_original=False,
        )
        transformer.fit(input_df)
        transformer.transform(X=input_df)


@pytest.mark.parametrize(
    'input_data',
    [
        {
            'a_latitude': [0, 0],
            'a_longitude': [None, 0],
            'b_latitude': [0, 0],
            'b_longitude': [0, 0],
        },
    ]
)
def test_fit_raises_error_if_na_in_df(input_data):
    input_df = pd.DataFrame(input_data)
    with pytest.raises(ValueError):
        transformer = DistanceFeatures(
            a_latitude='a_latitude',
            a_longitude='a_longitude',
            b_latitude='b_latitude',
            b_longitude='b_longitude',
            output_column_name='distance_between_a_and_b',
            drop_original=False,
        )
        transformer.fit(input_df)
        transformer.transform(X=input_df)


@pytest.mark.parametrize(
    'input_data',
    [
        {
            'a_latitude': [0, 0],
            'a_longitude': [0, 0],
            'b_latitude': [0, 0],
            'b_longitude': [0, 0],
        },
    ]
)
def test_fit_raises_error_if_lat_lon_columns_not_in_df(input_data):
    input_df = pd.DataFrame(input_data)
    with pytest.raises(ValueError):
        transformer = DistanceFeatures(
            a_latitude='a_latitude',
            a_longitude='a_longitude',
            b_latitude='<wrong-name>',
            b_longitude='b_longitude',
            output_column_name='distance_between_a_and_b',
            drop_original=False,
        )
        transformer.fit(input_df)
        transformer.transform(X=input_df)


def test_raises_error_when_init_parameters_not_permitted():
    with pytest.raises(ValueError):
        DistanceFeatures(
            a_latitude='a_latitude',
            a_longitude='a_longitude',
            b_latitude='b_latitude',
            b_longitude='b_longitude',
            output_column_name='distance_between_a_and_b',
            drop_original='False',
        )

    with pytest.raises(ValueError):
        DistanceFeatures(
            a_latitude=123,
            a_longitude='a_longitude',
            b_latitude='b_latitude',
            b_longitude='b_longitude',
            output_column_name='distance_between_a_and_b',
            drop_original=False,
        )

    with pytest.raises(ValueError):
        DistanceFeatures(
            a_latitude='a_latitude',
            a_longitude='a_longitude',
            b_latitude='b_latitude',
            b_longitude='b_longitude',
            output_column_name=123,
            drop_original=False,
        )
