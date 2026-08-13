"""
Integration tests for state flow — reducer merges and node invocations.

Tests that:
- Reducer functions correctly merge state updates
- State fields initialise with expected types
- any_true / keep_first / merge_* reducers handle edge cases

These tests use idd_core reducers directly; no API key required.
"""
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def core():
    try:
        import idd_core
        return idd_core
    except ImportError:
        pytest.skip("idd_core.py not available")


class TestReducerStateFlow:
    """Verify reducer functions behave correctly as part of a state update cycle."""

    def test_keep_first_prefers_existing_value(self, core):
        """keep_first: existing non-None value must not be overwritten."""
        assert core.keep_first("original", "new") == "original"

    def test_keep_first_accepts_new_when_none(self, core):
        """keep_first: None existing value is replaced by new value."""
        assert core.keep_first(None, "new") == "new"

    def test_any_true_with_one_true(self, core):
        """any_true: returns True when any arg is truthy."""
        assert core.any_true(False, True) is True

    def test_any_true_both_false(self, core):
        """any_true: returns False when both args are falsy."""
        assert core.any_true(False, False) is False

    def test_any_true_existing_stays_true(self, core):
        """any_true: once True, stays True even if new value is False."""
        assert core.any_true(True, False) is True

    def test_merge_messages_concatenates(self, core):
        """add_messages (LangGraph built-in): appends new messages to existing list."""
        if not core.HAS_LANGCHAIN:
            pytest.skip("LangChain not available")
        from langchain_core.messages import HumanMessage
        existing = [HumanMessage(content="Hello")]
        new_msg = [HumanMessage(content="World")]
        result = core.add_messages(existing, new_msg)
        assert len(result) >= 2

    def test_dict_merge_shallow_new_wins(self, core):
        """dict_merge_shallow: new dict values overwrite existing."""
        result = core.dict_merge_shallow({"a": 1, "b": 2}, {"b": 99, "c": 3})
        assert result["b"] == 99
        assert result["a"] == 1
        assert result["c"] == 3

    def test_dict_merge_shallow_none_old(self, core):
        """dict_merge_shallow: if old is None, returns new."""
        result = core.dict_merge_shallow(None, {"x": 1})
        assert result == {"x": 1}

    def test_reduce_plan_keep_sorted_merge(self, core):
        """_reduce_plan_keep_sorted: returns plan with merged steps from both plans."""
        base_fields = {
            "reply_msg_to_supervisor": "",
            "finished_this_task": False,
            "expect_reply": False,
        }
        step_a = {
            "step_number": 1, "step_name": "Step A", "step_description": "Do A",
            "is_step_complete": True, "plan_version": 1, **base_fields,
        }
        step_b = {
            "step_number": 2, "step_name": "Step B", "step_description": "Do B",
            "is_step_complete": False, "plan_version": 1, **base_fields,
        }
        plan_a = core.Plan.model_validate({
            "plan_version": 1, "plan_title": "Plan A", "plan_summary": "S",
            "plan_steps": [step_a], **base_fields,
        })
        plan_b = core.Plan.model_validate({
            "plan_version": 1, "plan_title": "Plan B", "plan_summary": "S",
            "plan_steps": [step_b], **base_fields,
        })
        merged = core._reduce_plan_keep_sorted(plan_a, plan_b)
        assert merged is not None
        step_nums = [s.step_number for s in merged.plan_steps]
        assert step_nums == sorted(step_nums), "Merged plan steps must be sorted"

    def test_reduce_plan_none_inputs(self, core):
        """_reduce_plan_keep_sorted: None inputs return the non-None plan."""
        base_fields = {
            "reply_msg_to_supervisor": "",
            "finished_this_task": False,
            "expect_reply": False,
        }
        plan = core.Plan.model_validate({
            "plan_version": 1, "plan_title": "P", "plan_summary": "S",
            "plan_steps": [], **base_fields,
        })
        assert core._reduce_plan_keep_sorted(None, plan) is plan
        assert core._reduce_plan_keep_sorted(plan, None) is plan
        assert core._reduce_plan_keep_sorted(None, None) is None
