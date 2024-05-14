def check_parameter_unseen(unseen, accepted_values):
    if not isinstance(accepted_values, list) or not all(
        isinstance(item, str) for item in accepted_values
    ):
        raise ValueError(
            "accepted_values should be a list of strings. "
            f" Got {accepted_values} instead."
        )
    if unseen not in accepted_values:
        raise ValueError(
            f"Parameter `unseen` takes only values {', '.join(accepted_values)}."
            f" Got {unseen} instead."
        )
