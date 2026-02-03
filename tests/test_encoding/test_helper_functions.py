import pytest

from feature_engine.encoding._helper_functions import check_parameter_unseen


@pytest.mark.parametrize("accepted", ["one", False, [1, 2], ("one", "two"), 1])
def test_raises_error_when_accepted_values_not_permitted(accepted):
    with pytest.raises(ValueError) as record:
        check_parameter_unseen("zero", accepted)
    msg = "accepted_values should be a list of strings. " f" Got {accepted} instead."
    assert str(record.value) == msg


@pytest.mark.parametrize("accepted", [["one", "two"], ["three", "four"]])
def test_raises_error_when_error_not_in_accepted_values(accepted):
    with pytest.raises(ValueError) as record:
        check_parameter_unseen("zero", accepted)
    msg = (
        f"Parameter `unseen` takes only values {', '.join(accepted)}."
        " Got zero instead."
    )
    assert str(record.value) == msg
