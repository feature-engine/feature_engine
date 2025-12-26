.. _geo_distance_transformer:

.. currentmodule:: feature_engine.creation

GeoDistanceTransformer
======================

The :class:`GeoDistanceTransformer()` calculates the distance between two geographical
coordinate pairs (latitude/longitude) and adds the result as a new feature.

This transformer is useful for location-based machine learning problems such as
real estate pricing, delivery route optimization, ride-sharing applications,
and any domain where geographic proximity is relevant.

Distance Methods
~~~~~~~~~~~~~~~~

The transformer supports different distance calculation methods:

- **haversine**: Great-circle distance using the Haversine formula (default).
  Most accurate for typical distances on Earth's surface.
- **euclidean**: Simple Euclidean distance in the coordinate space.
  Fast but less accurate for long distances.
- **manhattan**: Manhattan (taxicab) distance in coordinate space.
  Useful as a rough approximation for grid-based city layouts.

Output Units
~~~~~~~~~~~~

The distance can be output in various units:

- **km**: Kilometers (default)
- **miles**: Miles
- **meters**: Meters
- **feet**: Feet

Example
~~~~~~~

Let's create a dataframe with origin and destination coordinates:

.. code:: python

    import pandas as pd
    from feature_engine.creation import GeoDistanceTransformer

    # Sample data: trips between US cities
    X = pd.DataFrame({
        'origin_lat': [40.7128, 34.0522, 41.8781, 29.7604],
        'origin_lon': [-74.0060, -118.2437, -87.6298, -95.3698],
        'dest_lat': [34.0522, 41.8781, 40.7128, 33.4484],
        'dest_lon': [-118.2437, -87.6298, -74.0060, -112.0740],
        'trip_id': [1, 2, 3, 4]
    })

Now let's calculate the distances:

.. code:: python

    # Set up the transformer
    gdt = GeoDistanceTransformer(
        lat1='origin_lat',
        lon1='origin_lon',
        lat2='dest_lat',
        lon2='dest_lon',
        method='haversine',
        output_unit='km',
        output_col='distance_km'
    )

    # Fit and transform
    gdt.fit(X)
    X_transformed = gdt.transform(X)

    print(X_transformed[['trip_id', 'distance_km']])

Output:

.. code:: python

       trip_id   distance_km
    0        1   3935.746254
    1        2   2808.517344
    2        3   1144.286561
    3        4   1634.724892

Using different distance methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Euclidean distance (faster but less accurate)
    gdt_euclidean = GeoDistanceTransformer(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        method='euclidean',
        output_col='distance_euclidean'
    )

    # Manhattan distance (useful for grid cities)
    gdt_manhattan = GeoDistanceTransformer(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        method='manhattan',
        output_col='distance_manhattan'
    )

Converting to miles
~~~~~~~~~~~~~~~~~~~

.. code:: python

    gdt = GeoDistanceTransformer(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        output_unit='miles',
        output_col='distance_miles'
    )

    gdt.fit(X)
    X_transformed = gdt.transform(X)

Dropping original coordinate columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    gdt = GeoDistanceTransformer(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        drop_original=True
    )

    gdt.fit(X)
    X_transformed = gdt.transform(X)

    # Coordinate columns are removed
    print(X_transformed.columns.tolist())
    # ['trip_id', 'geo_distance']

Using in a Pipeline
~~~~~~~~~~~~~~~~~~~

:class:`GeoDistanceTransformer()` works seamlessly with scikit-learn pipelines:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    # Create a pipeline for price prediction
    pipe = Pipeline([
        ('geo_distance', GeoDistanceTransformer(
            lat1='origin_lat', lon1='origin_lon',
            lat2='dest_lat', lon2='dest_lon',
            output_unit='km',
            drop_original=True
        )),
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor())
    ])

API Reference
-------------

.. autoclass:: GeoDistanceTransformer
    :members:
    :inherited-members:
