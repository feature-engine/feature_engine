.. _information_value:

.. currentmodule:: feature_engine.selection

SelectByInformationValue
========================

:class:`SelectByInformationValue()` selects features based on whether the feature's information value is
greater than the threshold passed by the user. The transformer is only compatible with categorical
features.

Information value (IV) is used to assess a categorical feature's predictive power of a binary-class dependent
variable. To derive a feature's IV, the weight of evidence (WoE) must first be calculated for each
unique category or bin that comprises the feature. If a category or bin contains a large percentage
of true or positive labels compared to the percentage of false or negative labels, then that category
or bin will have a high WoE value.

:class:`WoE()` is used to calcaluate the WoE values for each category.

Once the WoE is derived, :class:`SelectByInformationValue()` calculates the information value (IV)
for each unique category or bin. The transformer than sums the individual IVs for each category providing
the IV score for the feature. This value assesses the feature's predictive power in capturing the binary
dependent variable.

The table below presents a general rule-of-thumb for using IV to determine a variable's predictive power:

.. list-table::
    :widths: 30 30
    :header-rows: 1

    * - Information Value
      - Predictive Power
    * - < 0.02
      - Useless
    * - 0.02 to 0.1
      - Weak
    * - 0.1 to 0.3
      - Medium
    * - 0.3 to 0.5
      - Strong
    * - > 0.5
      - Suspicious, too good to be true

.. table::
    :align: left

+----------------------+---------------------------------+
| Information Value    | Predictive Power                |
+======================+=================================+
| < 0.02               | Useless                         |
+----------------------+---------------------------------+
| 0.02 to 0.1          | Weak                            |
+----------------------+---------------------------------+
| 0.1 to 0.3           | Medium                          |
+----------------------+---------------------------------+
| 0.3 to 0.5           | Strong                          |
+----------------------+---------------------------------+
| > 0.5                | Suspicious, too good to be true |
+----------------------+---------------------------------+




