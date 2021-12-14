.. -*- mode: rst -*-

Datetime Features
=================

Oftentimes datasets contain information related to what point in time
observations occurred. Be it natively labelled as datetime or, more generically,
as object-like, datatime variables usually come in a far from ideal format to be
further analyzed and engineered. 
The DatetimeFeatures transformer contained in this module is able to extract
many different numerical and binary datetime features from your dataset, such
as which month an observation occurred, which day of the week, or whether that day
was a weekend day or not. Granting you the ability to choose exactly which features
to extract from a rather wide pool, and which columns of your dataset to extract
them from, DatetimeFeatures is built to significantly reduce the amount of duplicated
and boilerplate code when it comes to dealing with datetime variables.

**Datetime transformers**

.. toctree::
   :maxdepth: 2

   DatetimeFeatures