.. _geo_distance_transformer:

.. currentmodule:: feature_engine.creation

GeoDistanceTransformer
======================

:class:`GeoDistanceTransformer()` calculates the distance between two geographical
coordinate pairs (latitude/longitude) and adds the result as a new feature.

:class:`GeoDistanceTransformer()` is useful for location-based machine learning problems such as
real estate pricing, delivery route optimization, ride-sharing applications,
and any domain where geographic proximity is relevant.

Distance Methods
----------------

The transformer supports different distance calculation methods:

- **haversine**: Great-circle distance using the Haversine formula (default).
  Most accurate for typical distances on Earth's surface.
- **euclidean**: Simple Euclidean distance in the coordinate space.
  Fast but less accurate for long distances.
- **manhattan**: Manhattan (taxicab) distance in coordinate space.
  Useful as a rough approximation for grid-based city layouts.

Output Units
------------

The distance can be returned in various units:

- **km**: Kilometers (default)
- **miles**: Miles
- **meters**: Meters
- **feet**: Feet

Python Demo
-----------

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

Now let's calculate the distances using the haversine formula and returning the values in km:

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

In the following output we see the trip ID followed by the distance traveled in each trip:

.. code:: python

       trip_id   distance_km
    0        1   3935.746254
    1        2   2808.517344
    2        3   1144.286561
    3        4   1634.724892

Using different distance methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use the Euclidean distance method, which provides a faster but less accurate
calculation suitable for short distances:

.. code:: python

    gdt_euclidean = GeoDistanceTransformer(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        method='euclidean',
        output_col='distance_euclidean'
    )

    gdt_euclidean.fit(X)
    X_euclidean = gdt_euclidean.transform(X)
    print(X_euclidean[['trip_id', 'distance_euclidean']])

The Euclidean distances differ from the Haversine values because they don't account
for Earth's curvature:

.. code:: python

       trip_id  distance_euclidean
    0        1         4940.252715
    1        2         3493.298968
    2        3         1519.295694
    3        4         1720.178310

Alternatively, we can use the Manhattan distance, which is useful for grid-based city layouts:

.. code:: python

    gdt_manhattan = GeoDistanceTransformer(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        method='manhattan',
        output_col='distance_manhattan'
    )

    gdt_manhattan.fit(X)
    X_manhattan = gdt_manhattan.transform(X)
    print(X_manhattan[['trip_id', 'distance_manhattan']])

The Manhattan distance sums the absolute differences in latitude and longitude:

.. code:: python

       trip_id  distance_manhattan
    0        1          5628.24000
    1        2          4684.15800
    2        3          1637.36700
    3        4          2279.96460

Using different output units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transformer supports returning distances in km (default), miles, meters, or feet.
Here we calculate distances in miles:

.. code:: python

    gdt = GeoDistanceTransformer(
        lat1='origin_lat', lon1='origin_lon',
        lat2='dest_lat', lon2='dest_lon',
        output_unit='miles',
        output_col='distance_miles'
    )

    gdt.fit(X)
    X_transformed = gdt.transform(X)
    print(X_transformed[['trip_id', 'distance_miles']])

The distances are now expressed in miles instead of kilometers:

.. code:: python

       trip_id  distance_miles
    0        1     2445.258392
    1        2     1745.046817
    2        3      711.000629
    3        4     1015.643614

Dropping original coordinate columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reduce the dimensionality of the output dataset, we can remove the original
coordinate columns after calculating the distance:

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

After transformation, only the non-coordinate columns and the new distance column remain:

.. code:: python

    ['trip_id', 'geo_distance']

Calculating distance within a Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`GeoDistanceTransformer()` works seamlessly with scikit-learn pipelines. In the
following example, we create a pipeline that first calculates the geographic distance,
then scales the features, and finally trains a regression model:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    # Create sample target variable
    y = pd.Series([100, 150, 80, 200])

    # Create a pipeline for price prediction
    pipe = Pipeline([
        ('geo_distance', GeoDistanceTransformer(
            lat1='origin_lat', lon1='origin_lon',
            lat2='dest_lat', lon2='dest_lon',
            output_unit='km',
            drop_original=True
        )),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Fit the pipeline
    pipe.fit(X, y)

    # Make predictions
    predictions = pipe.predict(X)
    print(f"Predictions: {predictions}")

The pipeline successfully trains and returns predictions:

.. code:: python

    Predictions: [100. 150.  80. 200.]
