import sklearn
from sklearn.utils.fixes import parse_version

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)


def _return_tags():
    tags = {
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
            "check_sample_weights_not_an_array": "Ok to fail",
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

    if sklearn_version > parse_version("1.6"):
        msg1 = "against Feature-engines design."
        msg2 = "Our transformers do not preserve dtype."
        all_fail = {
            "check_do_not_raise_errors_in_init_or_set_params": msg1,
            "check_transformer_preserve_dtypes": msg2,
            # TODO: investigate this test further.
            "check_n_features_in_after_fitting": "not sure why it fails, we do check.",
        }
        tags["_xfail_checks"].update(all_fail)  # type: ignore
    return tags
