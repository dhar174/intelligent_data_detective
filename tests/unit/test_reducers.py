"""
Unit tests for IDD reducer functions.

Reducers tested:
- keep_first: preserves first non-None value
- dict_merge_shallow: shallow merge with new-wins
- merge_lists / merge_unique
- merge_int_sum
- merge_dicts / merge_dict
- any_true
- last_wins
- _reduce_plan_keep_sorted (covered in test_models.py; light coverage here)
"""
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def core():
    try:
        import idd_core
        return idd_core
    except ImportError:
        pytest.skip("idd_core.py not available")


class TestKeepFirst:
    def test_a_set_b_set(self, core):
        assert core.keep_first("first", "second") == "first"

    def test_a_none_returns_b(self, core):
        assert core.keep_first(None, "second") == "second"

    def test_a_set_b_none(self, core):
        assert core.keep_first("first", None) == "first"

    def test_both_none(self, core):
        assert core.keep_first(None, None) is None

    def test_zero_is_truthy_enough(self, core):
        """0 is not None — should be preserved as 'first'."""
        assert core.keep_first(0, 99) == 0

    def test_empty_string_preserved(self, core):
        assert core.keep_first("", "b") == ""

    def test_false_preserved(self, core):
        assert core.keep_first(False, True) is False


class TestDictMergeShallow:
    def test_both_none(self, core):
        assert core.dict_merge_shallow(None, None) == {}

    def test_old_none(self, core):
        assert core.dict_merge_shallow(None, {"a": 1}) == {"a": 1}

    def test_new_none(self, core):
        assert core.dict_merge_shallow({"a": 1}, None) == {"a": 1}

    def test_new_overrides_old(self, core):
        result = core.dict_merge_shallow({"a": 1, "b": 2}, {"b": 99, "c": 3})
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_no_mutation_of_inputs(self, core):
        old = {"x": 1}
        new = {"y": 2}
        result = core.dict_merge_shallow(old, new)
        assert old == {"x": 1}
        assert new == {"y": 2}
        assert result == {"x": 1, "y": 2}


class TestMergeLists:
    def test_two_lists(self, core):
        assert core.merge_lists([1, 2], [3, 4]) == [1, 2, 3, 4]

    def test_a_none(self, core):
        assert core.merge_lists(None, [1, 2]) == [1, 2]

    def test_b_none(self, core):
        assert core.merge_lists([1, 2], None) == [1, 2]

    def test_both_none(self, core):
        assert core.merge_lists(None, None) == []

    def test_preserves_duplicates(self, core):
        assert core.merge_lists([1, 1], [1, 2]) == [1, 1, 1, 2]


class TestMergeUnique:
    def test_deduplicates(self, core):
        result = core.merge_unique(["a", "b"], ["b", "c"])
        assert result == ["a", "b", "c"]

    def test_preserves_order(self, core):
        result = core.merge_unique(["c", "a"], ["b"])
        assert result == ["c", "a", "b"]

    def test_none_handling(self, core):
        assert core.merge_unique(None, ["x"]) == ["x"]
        assert core.merge_unique(["x"], None) == ["x"]
        assert core.merge_unique(None, None) == []


class TestMergeIntSum:
    def test_basic_sum(self, core):
        assert core.merge_int_sum(3, 4) == 7

    def test_none_a(self, core):
        assert core.merge_int_sum(None, 5) == 5

    def test_none_b(self, core):
        assert core.merge_int_sum(5, None) == 5

    def test_both_none(self, core):
        assert core.merge_int_sum(None, None) == 0

    def test_zero_plus_zero(self, core):
        assert core.merge_int_sum(0, 0) == 0


class TestAnyTrue:
    def test_both_true(self, core):
        assert core.any_true(True, True) is True

    def test_one_true(self, core):
        assert core.any_true(True, False) is True
        assert core.any_true(False, True) is True

    def test_both_false(self, core):
        assert core.any_true(False, False) is False

    def test_none_is_falsy(self, core):
        assert core.any_true(None, None) is False
        assert core.any_true(True, None) is True


class TestLastWins:
    def test_returns_b(self, core):
        assert core.last_wins("old", "new") == "new"

    def test_none_b(self, core):
        assert core.last_wins("old", None) is None

    def test_none_a(self, core):
        assert core.last_wins(None, "new") == "new"


class TestMergeDicts:
    def test_merge_dicts(self, core):
        result = core.merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_dict(self, core):
        result = core.merge_dict({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_new_wins(self, core):
        result = core.merge_dict({"a": 1}, {"a": 99})
        assert result["a"] == 99

    def test_none_handling(self, core):
        assert core.merge_dict(None, {"a": 1}) == {"a": 1}
        assert core.merge_dict({"a": 1}, None) == {"a": 1}
        assert core.merge_dict(None, None) == {}
