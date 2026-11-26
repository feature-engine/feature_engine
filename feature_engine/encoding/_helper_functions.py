import narwhals as nw
import narwhals.dependencies as nwd

TARGET_NAME = "__feature_engine_target__"


def check_parameter_unseen(unseen, accepted_values):
    if not isinstance(accepted_values, list) or not all(
        isinstance(item, str) for item in accepted_values
    ):
        raise ValueError(
            "accepted_values should be a list of strings. "
            f" Got {accepted_values} instead."
        )
    if not isinstance(unseen, str) or unseen not in accepted_values:
        raise ValueError(
            f"Parameter `unseen` takes only values {', '.join(accepted_values)}."
            f" Got {unseen} instead."
        )


def add_target_to_X(nw_X, y):
    """Add y to X as the column TARGET_NAME, pairing rows by position.

    y can be a series, list or array. With pandas, the column takes the index of X.
    """
    if nwd.is_into_series(y):
        y_nw = nw.from_native(y, series_only=True)
    else:
        y_nw = nw.new_series(name=TARGET_NAME, values=y, backend=nw_X.implementation)
    return nw_X.with_columns(y_nw.alias(TARGET_NAME))
