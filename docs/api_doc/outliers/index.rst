.. -*- mode: rst -*-

.. currentmodule:: feature_engine.outliers

Outlier Handling
================

Feature-engine's outlier transformers cap maximum or minimum values of a variable at an
arbitrary or derived value. The OutlierTrimmer removes outliers from the dataset.

=================================== ==============================================================
 Transformer                          Description
=================================== ==============================================================
:class:`Winsorizer()`                 Caps variables at automatically determined extreme values
:class:`ArbitraryOutlierCapper()`     Caps variables at values determined by the user
:class:`OutlierTrimmer()`             Removes outliers from the dataframe
=================================== ==============================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   Winsorizer
   ArbitraryOutlierCapper
   OutlierTrimmer