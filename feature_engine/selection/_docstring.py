_variables_all_docstring = """variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset.
    """.rstrip()

_variables_numerical_docstring = """variables: str or list, default=None
        The list of variable to be evaluated. If None, the transformer will evaluate
        all numerical features in the dataset.
        """.rstrip()

_variables_attribute_docstring = """variables_:
        The variables that will be considered for the feature selection.
        """.rstrip()

_missing_values_docstring = """missing_values: str, default=ignore
        Takes values 'raise' and 'ignore'. Whether the missing values should be raised
        as error or ignored when determining correlation.
        """.rstrip()

_estimator_docstring = """estimator: object
        A Scikit-learn estimator for regression or classification.
        """.rstrip()

_scoring_docstring = """scoring: str, default='roc_auc'
        Desired metric to optimise the performance of the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html
        """.rstrip()

_threshold_docstring = """threshold: float, int, default = 0.01
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold can be a big number.
        The threshold must be defined by the user. Bigger thresholds will select less
        features.
        """.rstrip()

_cv_docstring = """cv: int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use cross_validate's default 5-fold cross validation
            - int, to specify the number of folds in a (Stratified)KFold,
            - CV splitter
                - (https://scikit-learn.org/stable/glossary.html#term-CV-splitter)
            - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and y is either binary or
        multiclass, StratifiedKFold is used. In all other cases, KFold is used. These
        splitters are instantiated with `shuffle=False` so the splits will be the same
        across calls. For more details check Scikit-learn's `cross_validate`'s
        documentation.
        """.rstrip()

_initial_model_performance_docstring = """initial_model_performance_:
        Performance of the model trained using the original dataset.
        """.rstrip()

_features_to_drop_docstring = """features_to_drop_:
        List with the features to remove from the dataset.
        """.rstrip()

_fit_docstring = """fit:
        Find the important features.
        """.rstrip()

_transform_docstring = """transform:
        Reduce X to the selected features.
        """.rstrip()
