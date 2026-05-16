import pytest
from helpers.diffusion_utils import drop_condition


def test_drop_ratio_zero_no_change():
    cond = ["a", "b", "c", "d"]
    result = drop_condition(cond, 0.0)
    assert result == cond


def test_drop_ratio_one_all_empty():
    cond = ["a", "b", "c", "d"]
    result = drop_condition(cond, 1.0)
    assert all(s == "" for s in result)


def test_drop_ratio_half_correct_count():
    cond = ["a", "b", "c", "d"]
    result = drop_condition(cond, 0.5)
    num_dropped = sum(1 for s in result if s == "")
    assert num_dropped == 2  # int(4 * 0.5)


def test_length_preserved():
    cond = ["x"] * 10
    result = drop_condition(cond, 0.3)
    assert len(result) == len(cond)


def test_does_not_modify_original():
    cond = ["a", "b", "c"]
    original = cond[:]
    drop_condition(cond, 0.5)
    assert cond == original


def test_non_dropped_strings_unchanged():
    cond = ["hello", "world", "foo", "bar"]
    result = drop_condition(cond, 0.5)
    non_empty = [s for s in result if s != ""]
    assert all(s in cond for s in non_empty)


def test_ratio_above_one_raises():
    with pytest.raises(AssertionError):
        drop_condition(["a", "b"], 1.5)
