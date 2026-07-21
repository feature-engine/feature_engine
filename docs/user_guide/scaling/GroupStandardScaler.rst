.. _group_standard_scaler:

.. currentmodule:: feature_engine.scaling

GroupStandardScaler
===================

:class:`GroupStandardScaler()` scales numerical variables relative to a group. It standardises variables by removing the mean and scaling to unit variance per group. This means that for each group within the reference column, the scaler learns the mean and standard deviation of each variable to be scaled. During transformation, it applies the standard z-score formula.

The :class:`GroupStandardScaler()` requires numerical variables to be scaled, and at least one reference variable which acts as the grouping key.

Python example
--------------

We'll show how to use :class:`GroupStandardScaler()` through a toy dataset. Let's create a toy dataset:

.. code:: python

    import pandas as pd
    from feature_engine.scaling import GroupStandardScaler

    df = pd.DataFrame({
        "House_Price": [100000, 150000, 120000, 500000, 550000, 480000],
        "Neighborhood": ["A", "A", "A", "B", "B", "B"]
    })

    print(df)

The dataset looks like this:

.. code:: python

       House_Price Neighborhood
    0       100000            A
    1       150000            A
    2       120000            A
    3       500000            B
    4       550000            B
    5       480000            B

We want to scale the prices relative to the neighborhood so we know if a house is relatively expensive for its neighborhood.

.. code:: python

    # set up the scaler
    scaler = GroupStandardScaler(
        variables=["House_Price"],
        reference=["Neighborhood"]
    )

    # fit the scaler
    scaler.fit(df)

The scaler learns the mean and standard deviation of the House_Price per neighborhood:

.. code:: python

    print(scaler.barycenter_)
    # Means: {'House_Price': {'A': 123333.33333333333, 'B': 510000.0}}

    print(scaler.scale_)
    # Std Devs: {'House_Price': {'A': 25166.11478423583, 'B': 36055.51275463989}}

Now we can apply the transformation:

.. code:: python

    df_scaled = scaler.transform(df)
    print(df_scaled)

.. code:: python

       House_Price Neighborhood
    0    -0.927172            A
    1     1.059626            A
    2    -0.132453            A
    3    -0.277349            B
    4     1.109312            B
    5    -0.831963            B
