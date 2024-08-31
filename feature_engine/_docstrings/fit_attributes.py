"""Docstrings for the attributes that are generated during fit."""

_variables_attribute_docstring = """variables_:
        The group of variables that will be transformed.
        """.rstrip()

_feature_names_in_docstring = """feature_names_in_:
        List with the names of features seen during `fit`.
        """.rstrip()

_n_features_in_docstring = """n_features_in_:
        The number of features in the train set used in fit.
        """.rstrip()

# used by discretisers
_binner_dict_docstring = """binner_dict_:
         Dictionary with the interval limits per variable.
     """.rstrip()

# used by imputers
_imputer_dict_docstring = """imputer_dict_:
        Dictionary with the values to replace missing data in each variable.
    """.rstrip()

# used by outlier module
_right_tail_caps_docstring = """right_tail_caps_:
        Dictionary with the maximum values beyond which a value will be considered an
        outlier.
    """.rstrip()

_left_tail_caps_docstring = """left_tail_caps_:
        Dictionary with the minimum values beyond which a value will be considered an
        outlier.
    """.rstrip()

# used by selection module
_feature_importances_docstring = """feature_importances_:
        Pandas Series with the feature importance (comes from step 2)
    """.rstrip()

_feature_importances_std_docstring = """feature_importances_std_:
        Pandas Series with the standard deviation of the feature importance.
    """.rstrip()

_performance_drifts_docstring = """performance_drifts_:
        Dictionary with the performance drift per examined feature (comes from step 5).
    """.rstrip()

_performance_drifts_std_docstring = """performance_drifts_std_:
        Dictionary with the performance drift's standard deviation of the
        examined feature (comes from step 5).
    """.rstrip()
