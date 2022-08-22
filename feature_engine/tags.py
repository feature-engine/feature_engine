def _return_tags():
    return {
        "preserves_dtype": [],
        "_xfail_checks": {
            # Complex data in math terms, are values like 4i (imaginary numbers
            # so to speak). I've never seen such a thing in the dfs I've
            # worked with, so I don't think we need this test.
            "check_complex_data": "Test not needed.",
            # check that estimators treat dtype object as numeric if possible
            "check_dtype_object": "Feature-engine transformers use dtypes to select "
            "between numerical and categorical variables. Feature-engine trusts the "
            "user casts the variables appropriately",
            # Test fails because FE does not like the sklearn class _NotAnArray
            # The test aims to check that the check_X_y function from sklearn is
            # working, but we do not use that check, because we work with dfs.
            "check_transformer_data_not_an_array": "Ok to fail",
            # TODO: we probably need the test below!!
            "check_methods_sample_order_invariance": "Test does not work on dataframes",
            # TODO: we probably need the test below!!
            # the test below tests that a second fit overrides a first fit.
            # the problem is that the test does not work with pandas df.
            "check_fit_idempotent": "Test does not work on dataframes.",
            "check_fit2d_predict1d": "Test not relevant, Feature-engine transformers "
            "only work with dataframes.",
        },
    }
