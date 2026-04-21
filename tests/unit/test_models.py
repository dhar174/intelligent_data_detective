"""
Unit tests for Plan, PlanStep, CompletedStepsAndTasks.

Key invariants tested:
- Plan.plan_steps are sorted ascending by step_number after construction
- Plan auto-assigns plan_version via ClassVar counter (monotonically increasing)
- CompletedStepsAndTasks._inject_and_dedupe deduplication preserves sort order (bug regression test)
- _reduce_plan_keep_sorted merges two Plans correctly
"""
import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def core():
    try:
        import idd_core
        return idd_core
    except ImportError:
        pytest.skip("idd_core.py not available")


_BASE_FIELDS = {
    "reply_msg_to_supervisor": "",
    "finished_this_task": False,
    "expect_reply": False,
}


def make_step(step_number, name, description, is_complete=False, version=1):
    """Helper to build a PlanStep dict with all required fields."""
    return {
        "step_number": step_number,
        "step_name": name,
        "step_description": description,
        "is_step_complete": is_complete,
        "plan_version": version,
        **_BASE_FIELDS,
    }


def make_plan(core, steps=None, title="Test Plan", summary="summary", version=None):
    """Helper to build a Plan with optional explicit version."""
    if steps is None:
        steps = [
            make_step(1, "Step A", "Do A"),
            make_step(2, "Step B", "Do B"),
        ]
    # If steps are dicts without base fields, inject them
    enriched = []
    for s in steps:
        if isinstance(s, dict) and "reply_msg_to_supervisor" not in s:
            enriched.append({**s, **_BASE_FIELDS})
        else:
            enriched.append(s)
    data = {
        "plan_title": title,
        "plan_summary": summary,
        "plan_steps": enriched,
        "plan_version": version or 1,
        **_BASE_FIELDS,
    }
    return core.Plan.model_validate(data)


class TestPlanStep:
    def test_valid_plan_step(self, core):
        s = core.PlanStep(
            step_number=1,
            step_name="Analyze",
            step_description="Do analysis",
            is_step_complete=False,
            plan_version=1,
            reply_msg_to_supervisor="",
            finished_this_task=False,
            expect_reply=False,
        )
        assert s.step_number == 1

    def test_extra_fields_forbidden(self, core):
        with pytest.raises(ValidationError):
            core.PlanStep(
                step_number=1, step_name="x", step_description="y",
                is_step_complete=False, plan_version=1,
                reply_msg_to_supervisor="", finished_this_task=False, expect_reply=False,
                unknown="bad",
            )


class TestPlan:
    def test_plan_steps_sorted_after_construction(self, core):
        """Steps provided out of order must be sorted ascending by step_number."""
        steps = [
            {"step_number": 3, "step_name": "C", "step_description": "Do C",
             "is_step_complete": False, "plan_version": 1},
            {"step_number": 1, "step_name": "A", "step_description": "Do A",
             "is_step_complete": False, "plan_version": 1},
            {"step_number": 2, "step_name": "B", "step_description": "Do B",
             "is_step_complete": False, "plan_version": 1},
        ]
        plan = make_plan(core, steps=steps)
        nums = [s.step_number for s in plan.plan_steps]
        assert nums == sorted(nums), f"Expected sorted, got {nums}"

    def test_plan_version_auto_assigned(self, core):
        """Plan auto-assigns a monotonically increasing plan_version."""
        plan1 = make_plan(core, title="Plan One")
        plan2 = make_plan(core, title="Plan Two")
        # Both must have plan_version >= 1 and plan2 > plan1
        assert plan1.plan_version >= 1
        assert plan2.plan_version > plan1.plan_version

    def test_duplicate_step_numbers_raise(self, core):
        """Two steps with the same step_number must raise ValidationError."""
        steps = [
            {"step_number": 1, "step_name": "A", "step_description": "Do A",
             "is_step_complete": False, "plan_version": 1},
            {"step_number": 1, "step_name": "B", "step_description": "Do B",
             "is_step_complete": False, "plan_version": 1},
        ]
        with pytest.raises(ValidationError):
            make_plan(core, steps=steps)

    def test_all_steps_get_plan_version(self, core):
        """All plan steps must have the same plan_version as the Plan."""
        plan = make_plan(core)
        for step in plan.plan_steps:
            assert step.plan_version == plan.plan_version

    def test_empty_plan_steps_allowed(self, core):
        """Empty plan_steps list is valid."""
        plan = make_plan(core, steps=[])
        assert plan.plan_steps == []


class TestCompletedStepsAndTasks:
    def _make_completed(self, core, steps, plan=None):
        data = {
            "completed_steps": steps,
            "finished_tasks": ["task_1"],
            "progress_report": {
                "latest_progress": "All good.",
                "reply_msg_to_supervisor": "",
                "finished_this_task": False,
                "expect_reply": False,
            },
            "reply_msg_to_supervisor": "",
            "finished_this_task": False,
            "expect_reply": False,
        }
        ctx = {"plan": plan} if plan else {}
        return core.CompletedStepsAndTasks.model_validate(data, context=ctx)

    def test_inject_and_dedupe_preserves_sort_order(self, core):
        """
        REGRESSION TEST for the _inject_and_dedupe bug.
        Steps given out of order must come out sorted by step_number.
        Previously returned list(seen.values()) which ignored the sort.
        Fixed: returns dedup_list (sorted).
        """
        steps = [
            make_step(3, "C", "Do C", is_complete=True),
            make_step(1, "A", "Do A", is_complete=True),
            make_step(2, "B", "Do B", is_complete=True),
        ]
        obj = self._make_completed(core, steps)
        nums = [s.step_number for s in obj.completed_steps]
        assert nums == [1, 2, 3], f"Expected [1,2,3], got {nums} (sort bug still present?)"

    def test_duplicate_steps_deduplicated(self, core):
        """Exact duplicate steps (same triplet) must be deduplicated to one."""
        step = make_step(1, "A", "Do A", is_complete=True)
        obj = self._make_completed(core, [step, step])
        assert len(obj.completed_steps) == 1

    def test_incomplete_step_raises(self, core):
        """_sorted_no_dups_and_subset requires all steps to be is_step_complete=True."""
        steps = [make_step(1, "A", "Do A", is_complete=False)]
        with pytest.raises(ValidationError):
            self._make_completed(core, steps)

    def test_out_of_order_steps_raise_after_inject(self, core):
        """If steps can't be sorted (duplicate step_numbers with diff names), validator catches it."""
        steps = [
            make_step(2, "X", "X", is_complete=True),
            make_step(2, "Y", "Y", is_complete=True),
        ]
        # Two different steps with same step_number — dedup by triplet keeps both,
        # _sorted_no_dups_and_subset detects duplicate step_number and raises
        with pytest.raises(ValidationError):
            self._make_completed(core, steps)

    def test_plan_version_injected_from_context(self, core):
        """When plan is in context, plan_version is injected into steps."""
        plan = make_plan(core, steps=[make_step(1, "Step A", "Do A"), make_step(2, "Step B", "Do B")])
        steps = [make_step(1, "Step A", "Do A", is_complete=True, version=99)]
        obj = self._make_completed(core, steps, plan=plan)
        assert obj.completed_steps[0].plan_version == plan.plan_version


class TestReducePlanKeepSorted:
    def test_merge_two_plans(self, core):
        plan_a = make_plan(core, steps=[make_step(1, "A", "Do A", is_complete=True)])
        plan_b = make_plan(core, steps=[make_step(2, "B", "Do B")])
        merged = core._reduce_plan_keep_sorted(plan_a, plan_b)
        assert merged is not None
        nums = [s.step_number for s in merged.plan_steps]
        assert nums == sorted(nums)

    def test_none_a_returns_b(self, core):
        plan = make_plan(core)
        result = core._reduce_plan_keep_sorted(None, plan)
        assert result is plan

    def test_none_b_returns_a(self, core):
        plan = make_plan(core)
        result = core._reduce_plan_keep_sorted(plan, None)
        assert result is plan

    def test_both_none_returns_none(self, core):
        result = core._reduce_plan_keep_sorted(None, None)
        assert result is None

    def test_later_plan_step_wins_on_same_step_number(self, core):
        """When both plans have step_number=1, b's version wins (last-wins)."""
        plan_a = make_plan(core, steps=[make_step(1, "Old Name", "Old")])
        plan_b = make_plan(core, steps=[make_step(1, "New Name", "New", is_complete=True)])
        merged = core._reduce_plan_keep_sorted(plan_a, plan_b)
        assert merged.plan_steps[0].step_name == "New Name"
