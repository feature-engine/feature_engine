_fit_not_learn_docstring = """fit:
        This transformer does not learn parameters.
        """.rstrip()

_fit_transform_docstring = """fit_transform:
        Fit to data, then transform it.

    get_feature_names_out:
        Get output feature names for transformation.

    get_params:
        Get parameters for this estimator.

    set_params:
        Set the parameters of this estimator.
        """.rstrip()

_get_feature_names_out_docstring = """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features: str, list, default=None
            If `None`, then the names of all the variables in the
            transformed dataset is returned. If list with feature
            names, the features in the list will be returned. This
            parameter exists mostly for compatibility with the
            Scikit-learn Pipeline.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """.rstrip()

_inverse_transform_docstring = """inverse_transform:
        Convert the data back to the original representation.
        """.rstrip()
