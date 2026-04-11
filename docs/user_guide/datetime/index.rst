.. _datetime_module:

Datetime Features
=================

Feature-engine’s datetime transformers extract a wide variety of date and time features
from datetime variables. Datetime variables can be cast as datetime or object.

Summary of Feature-engine’s creation transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

================================ ===============================================================================
    Transformer                     	  Description
================================ ===============================================================================
:class:`DatetimeFeatures()`	        Extracts features like day, month, year, hour, minute, second, and more.
:class:`DatetimeOrdinal()`          Recodes variable as time elapsed since a certain date.
:class:`DatetimeSubtraction()`	    Calculates time difference between 2 datetime variables.
================================ ===============================================================================

.. toctree::
   :maxdepth: 1

   DatetimeFeatures
   DatetimeOrdinal
   DatetimeSubtraction