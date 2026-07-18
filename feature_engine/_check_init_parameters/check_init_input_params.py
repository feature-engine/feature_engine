def _check_param_missing_values(missing_values):
    if missing_values not in ["raise", "ignore"]:
        raise ValueError(
            "missing_values takes only values 'raise' or 'ignore'. "
            f"Got {missing_values} instead."
        )


def _check_param_drop_original(drop_original):
    if not isinstance(drop_original, bool):
        raise ValueError(
            "drop_original takes only boolean values True and False. "
            f"Got {drop_original} instead."
        )

def _check_return_empty_is_bool(return_empty):
    if not isinstance(return_empty, bool):
        raise ValueError(
            "return_empty takes only boolean values True and False. "
            f"Got {return_empty} instead."
        )