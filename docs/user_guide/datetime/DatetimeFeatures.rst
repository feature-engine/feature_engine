.. _datetime_features:

.. currentmodule:: feature_engine.datetime

DatetimeFeatures
================

The :class:`DatetimeFeatures()` extracts several datetime features from datetime
variables. It works with variables whose original dtype is datetime, and also with
object-like and categorical variables, provided that they can be parsed into datetime
format. It *cannot* extract features from numerical variables.

Oftentimes datasets contain information related dates and/or times at which an event
occurred. In pandas dataframes, these datetime variables can be cast as datetime or,
more generically, as object.

Datetime variables in their raw format, are generally not suitable to train machine
learning models. Yet, an enormous amount of information can be extracted from them.

The :class:`DatetimeFeatures()` is able to extract many numerical and binary date and
time features from these datetime variables. Among these features we can find the month
in which an event occurred, the day of the week, or whether that day was a weekend day.

With the The :class:`DatetimeFeatures()` you can choose which date and time features
to extract from your datetime variables. You can also extract date and time features
from a subset of datetime variables in your data.

Examples
--------

Extract date features
~~~~~~~~~~~~~~~~~~~~~

Some example to extract a few day features


Extract time features
~~~~~~~~~~~~~~~~~~~~~

Some example to extract time features


Extract date and time features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Extract features from time-aware variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One example when utc=True and another example when the variable is a datetime
and it is localized for example to european zone, with uct=True and then utc=None.
So 3 examples :p

#TODO: most likely the below example you took from the jupyter notebook, right?
I would remove, and instead create the examples with toy dataframes like those in the
conf files in the tests. These examples are easy ones, so that users can make copy and
paste and kind of quickly learn how to use the class.


.. code:: python

    # import useful modules and the DatetimeFeatures class
    import pandas as pd
    import matplotlib.pyplot as plt
    from feature_engine.datetime import DatetimeFeatures

    # load the dataset
    data = pd.read_csv('KaggleV2-May-2016.csv')

    # we don't need the ds to be huge for the sake of this demo
    data.drop(np.random.choice(data.index, 100000, replace=False), inplace=True)

    # initialize class
    # we want to extract the features day of the week, month and hour from 
    # the variable *ScheduledDay*
    features_to_extract = ['day_of_the_week', 'month', 'hour']
    date_transformer = DatetimeFeatures(
        variables='ScheduledDay',
        features_to_extract=features_to_extract,
    )

    # plot the extracted features
    plt.figure(figsize=(20,6))
    for i,var in enumerate(["ScheduledDay_dotw", "ScheduledDay_month", "ScheduledDay_hour"]):
        plt.subplot(1, 3, i + 1)
        plt.title(var)
        data_transformed[var].value_counts().plot.bar()
    
.. image:: ../../images/datetime_extracted_features.png


More details
^^^^^^^^^^^^

You can find creative ways to use the :class:`DatetimeFeatures()` in the
following Jupyter notebook.

#TODO: change the link below to your notebook:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/creation/MathematicalCombination.ipynb>`_
