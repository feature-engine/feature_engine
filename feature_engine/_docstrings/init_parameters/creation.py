_features_to_combine = """features_to_combine: integer, list or tuple, default=None
        Used to determine how the variables indicated in `variables` will be combined
        to obtain the new features by using decision trees. If `integer`, then the value
        corresponds to the largest size of combinations allowed between features. For
        example, if you want to combine three variables, ["var_A", "var_B", "var_C"],
        and:

            - `features_to_combine = 1`, the transformer returns new features based on
                the predictions of a decision tree trained on each individual variable,
                generating 3 new features.

            - `features_to_combine = 2`, the transformer returns the features from
                `features_to_combine=1`, plus features based on the predictions of a
                decision tree based on all possible combinations of 2 variables, i.e.,
                ("var_A", "var_B"), ("var_A", "var_C"), and ("var_B", "var_C"),
                resulting in a total of 6 new features.

            - `features_to_combine = 3`, the transformer returns the features from
                `features_to_combine=2`, plus one additional feature based on the
                predictions of a decision trained on the 3 variables,
                ["var_A", "var_B", "var_C"], resulting in a total of 7 new features.

        If `list`, the list must contain integers indicating the number of features that
        should be used as input of a decision tree. For example, if the data has 4
        variables, ["var_A", "var_B", "var_C", "var_D"] and and
        `features_to_combine = [2,3]`, then all possible combinations of 2 and 3 v
        ariables will be returned. That'll result in the following combinations:
        ("var_A", "var_B"), ("var_A", "var_C"), ("var_A", "var_D"), ("var_B", "var_C"),
        ("var_B", "var_D"), ("var_C", "var_D"), ("var_A", "var_B", "var_C"),
        ("var_A", "var_B", "var_D"), ("var_A", "var_C", "var_D"), and
        ("var_B", "var_C", "var_D").\n
    |
        If `tuple`, the tuple must contain strings and/or tuples that indicate how to
        combine the variables to create the new features. For example, if
        `features_to_combine=("var_C", ("var_A", "var_C"), "var_C", ("var_B", "var_D")`,
        then, the transformer will train a decision tree based of each value within the
        tuple, resulting in 4 new features.\n
    |
        If `None`, then the transformer will create all possible combinations of 1 or
        more features, and use those as inputs to decision trees. This is equivalent to
        passing an integer that is equal to the number of variables to combine.\n
    """.rstrip()
