.. -*- mode: rst -*-

.. currentmodule:: feature_engine.wrappers

Scikit-learn Wrapper
====================

Feature-engine's Scikit-learn wrappers wrap Scikit-learn transformers allowing their
implementation only on a selected subset of features.

.. toctree::
   :maxdepth: 2

   Wrapper

The :class:`SklearnTransformerWrapper()` offers a similar function to the
`ColumnTransformer class <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_
available in Scikit-learn.