.. _information_value:

.. currentmodule:: feature_engine.selection

SelectByInformationValue
========================

:class:`SelectByInformationValue()` selects features based on whether the feature's information value is
greater than the threshold passed by the user.

Information value (IV) is used to assess a categorical feature's predictive power of a binary-class dependent
variable. To derive a feature's IV, the weight of evidence (WoE) must first be calculated for each
unique category or bin that comprises the feature. If a category or bin contains a large percentage
of true or positive labels compared to the percentage of false or negative labels, then that category
or bin will have a high WoE value. This signifies that that category or bin delineates between true
and negative labels.
