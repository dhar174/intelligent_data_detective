"""
Integration tests for state accumulation via reducer functions in idd_core.
Tests simulate multi-round LangGraph state updates by calling reducers directly
without importing the LangGraph State TypedDict (which requires LangChain).
"""
import pytest

pytestmark = [pytest.mark.integration]

from idd_core import (
    Plan,
    PlanStep,
    CompletedStepsAndTasks,
    ProgressReport,
    _reduce_plan_keep_sorted,
    any_true,
    last_wins,
    merge_int_sum,
    merge_lists,
    merge_unique,
)

BASE = dict(reply_msg_to_supervisor="test", finished_this_task=True, expect_reply=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_step(number, name="step", desc="description", complete=False):
    return PlanStep(
        step_number=number,
        step_name=f"{name}_{number}",
        step_description=desc,
        is_step_complete=complete,
        plan_version=1,
        **BASE,
    )


def make_plan(*step_numbers):
    steps = [make_step(n) for n in step_numbers]
    return Plan(
        plan_version=1,
        plan_title="Test Plan",
        plan_summary="Integration test plan",
        plan_steps=steps,
        **BASE,
    )


def make_progress(msg="in progress"):
    return ProgressReport(latest_progress=msg, **BASE)


# ---------------------------------------------------------------------------
# merge_lists
# ---------------------------------------------------------------------------

class TestMergeLists:
    def test_both_none_returns_empty_list(self):
        assert merge_lists(None, None) == []

    def test_none_plus_list_returns_list(self):
        assert merge_lists(None, [1, 2]) == [1, 2]

    def test_list_plus_none_returns_list(self):
        assert merge_lists([1, 2], None) == [1, 2]

    def test_concatenates_two_lists(self):
        assert merge_lists([1, 2], [3, 4]) == [1, 2, 3, 4]

    def test_preserves_duplicates(self):
        assert merge_lists([1, 1], [1]) == [1, 1, 1]

    def test_order_preserved_a_before_b(self):
        result = merge_lists(["a", "b"], ["c"])
        assert result.index("a") < result.index("c")


# ---------------------------------------------------------------------------
# merge_unique
# ---------------------------------------------------------------------------

class TestMergeUnique:
    def test_both_none_returns_empty_list(self):
        assert merge_unique(None, None) == []

    def test_deduplicates_overlap(self):
        result = merge_unique(["a", "b"], ["b", "c"])
        assert sorted(result) == ["a", "b", "c"]

    def test_preserves_insertion_order(self):
        result = merge_unique(["x", "y"], ["y", "z"])
        assert result == ["x", "y", "z"]

    def test_no_duplicates_in_result(self):
        result = merge_unique(["a", "a", "b"], ["b", "c"])
        assert result.count("a") == 1
        assert result.count("b") == 1


# ---------------------------------------------------------------------------
# merge_int_sum
# ---------------------------------------------------------------------------

class TestMergeIntSum:
    def test_both_none_returns_zero(self):
        assert merge_int_sum(None, None) == 0

    def test_int_plus_none_returns_int(self):
        assert merge_int_sum(5, None) == 5

    def test_none_plus_int_returns_int(self):
        assert merge_int_sum(None, 3) == 3

    def test_two_ints_summed(self):
        assert merge_int_sum(7, 3) == 10

    def test_accumulates_across_rounds(self):
        total = 0
        total = merge_int_sum(total, 5)
        total = merge_int_sum(total, 3)
        total = merge_int_sum(total, 2)
        assert total == 10


# ---------------------------------------------------------------------------
# any_true
# ---------------------------------------------------------------------------

class TestAnyTrue:
    def test_both_false_returns_false(self):
        assert any_true(False, False) is False

    def test_both_none_returns_false(self):
        assert any_true(None, None) is False

    def test_first_true_returns_true(self):
        assert any_true(True, False) is True

    def test_second_true_returns_true(self):
        assert any_true(False, True) is True

    def test_both_true_returns_true(self):
        assert any_true(True, True) is True


# ---------------------------------------------------------------------------
# last_wins
# ---------------------------------------------------------------------------

class TestLastWins:
    def test_returns_b(self):
        assert last_wins("a", "b") == "b"

    def test_b_overrides_a(self):
        assert last_wins(1, 2) == 2

    def test_b_none_overrides_a(self):
        assert last_wins("something", None) is None

    def test_a_none_returns_b(self):
        assert last_wins(None, "winner") == "winner"


# ---------------------------------------------------------------------------
# Plan creation: monotonic plan_version
# ---------------------------------------------------------------------------

class TestPlanVersionMonotonicity:
    def test_successive_plans_have_increasing_version(self, reset_plan_version_counter):
        plan1 = make_plan(1)
        plan2 = make_plan(2)
        plan3 = make_plan(3)
        assert plan2.plan_version > plan1.plan_version
        assert plan3.plan_version > plan2.plan_version

    def test_step_versions_synced_to_plan_version(self, reset_plan_version_counter):
        plan = make_plan(1, 2, 3)
        for step in plan.plan_steps:
            assert step.plan_version == plan.plan_version


# ---------------------------------------------------------------------------
# _reduce_plan_keep_sorted
# ---------------------------------------------------------------------------

class TestReducePlanKeepSorted:
    def test_merge_steps_sorted_by_step_number(self, reset_plan_version_counter):
        plan_a = make_plan(1, 3)
        plan_b = make_plan(2)
        merged = _reduce_plan_keep_sorted(plan_a, plan_b)
        nums = [s.step_number for s in merged.plan_steps]
        assert nums == sorted(nums)
        assert set(nums) == {1, 2, 3}

    def test_duplicate_step_number_b_wins(self, reset_plan_version_counter):
        plan_a = make_plan(1)
        plan_b = make_plan(1)
        # Force different names so we can tell which won
        plan_b.plan_steps[0].step_name = "step_1_from_b"
        merged = _reduce_plan_keep_sorted(plan_a, plan_b)
        assert len(merged.plan_steps) == 1
        assert merged.plan_steps[0].step_name == "step_1_from_b"

    def test_none_plus_plan_returns_plan(self, reset_plan_version_counter):
        plan = make_plan(1, 2)
        result = _reduce_plan_keep_sorted(None, plan)
        assert result is plan

    def test_plan_plus_none_returns_plan(self, reset_plan_version_counter):
        plan = make_plan(1)
        result = _reduce_plan_keep_sorted(plan, None)
        assert result is plan

    def test_both_none_returns_none(self):
        assert _reduce_plan_keep_sorted(None, None) is None


# ---------------------------------------------------------------------------
# CompletedStepsAndTasks
# ---------------------------------------------------------------------------

class TestCompletedStepsAndTasksAccumulation:
    def test_no_context_any_step_accepted(self, reset_plan_version_counter):
        step = make_step(99, name="orphan", complete=True)
        cst = CompletedStepsAndTasks(
            completed_steps=[step],
            finished_tasks=["task_a"],
            progress_report=make_progress(),
            **BASE,
        )
        assert len(cst.completed_steps) == 1

    def test_with_plan_context_valid_step_accepted(self, reset_plan_version_counter):
        plan = make_plan(1, 2, 3)
        step1_data = plan.plan_steps[0].model_dump()
        step1_data["is_step_complete"] = True  # completed_steps requires is_step_complete=True
        cst = CompletedStepsAndTasks.model_validate(
            dict(
                completed_steps=[step1_data],
                finished_tasks=["task_1"],
                progress_report=make_progress().model_dump(),
                **BASE,
            ),
            context={"plan": plan},
        )
        assert cst.completed_steps[0].step_number == 1

    def test_with_plan_context_invalid_step_raises(self, reset_plan_version_counter):
        from pydantic import ValidationError
        plan = make_plan(1, 2)
        out_of_plan_step = make_step(99, name="orphan")
        with pytest.raises(ValidationError):
            CompletedStepsAndTasks.model_validate(
                dict(
                    completed_steps=[out_of_plan_step.model_dump()],
                    finished_tasks=[],
                    progress_report=make_progress().model_dump(),
                    **BASE,
                ),
                context={"plan": plan},
            )

    def test_empty_completed_steps_valid(self, reset_plan_version_counter):
        cst = CompletedStepsAndTasks(
            completed_steps=[],
            finished_tasks=[],
            progress_report=make_progress("nothing done yet"),
            **BASE,
        )
        assert cst.completed_steps == []

    def test_multiple_rounds_accumulate_via_merge_lists(self, reset_plan_version_counter):
        plan = make_plan(1, 2, 3)
        step1 = plan.plan_steps[0]
        step2 = plan.plan_steps[1]

        round1 = [step1]
        round2 = [step2]
        accumulated = merge_lists(round1, round2)
        assert len(accumulated) == 2
        nums = {s.step_number for s in accumulated}
        assert nums == {1, 2}
