def check_parameter_unseen(errors, accepted_values):
    if errors not in accepted_values:
        raise ValueError(
            f"errors takes only values {', '.join(accepted_values)}."
            f"Got {errors} instead."
        )
