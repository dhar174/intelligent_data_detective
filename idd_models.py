"""
idd_models.py — Canonical Pydantic Models for Intelligent Data Detective.

Home of core planning models, data contracts, and reducers:
- BaseNoExtrasModel (strict extra="forbid" base contract)
- ProgressReport
- PlanStep
- Plan (lifecycle-aware: new plan allocation vs. persisted snapshot restoration)
- CompletedStepsAndTasks (RC-2 sorted deduplication and numeric step uniqueness)
- _reduce_plan_keep_sorted (state reducer for merging Plans)

Architecture: Checkpoint 1 (Issue #156 / Refs #140, #144, #145).
"""

from __future__ import annotations

import threading
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Base Model Contract
# ---------------------------------------------------------------------------

class BaseNoExtrasModel(BaseModel):
    """Base model enforcing extra="forbid" and standard agent response fields."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"additionalProperties": False},
    )

    reply_msg_to_supervisor: str = Field(
        ...,
        description=(
            "Message to send to the supervisor. Can be a simple message stating completion "
            "of the task, or it can be detailed information about the result, or you can put "
            "any questions for the supervisor here as well. This is ONLY for sending messages "
            "to the supervisor, NOT to worker agents. If you are the/a supervisor (or the "
            "router, planner, or progress reporter), this field should be empty unless you are "
            "expecting a reply from the main supervisor, NOT from a worker agent."
        ),
    )
    finished_this_task: bool = Field(
        ...,
        description=(
            "Whether this assigned task represented by this object has been completed. For "
            "example, if it is a Router object, this field should be True if the route "
            "decision has been made. Another example, if it is a CleaningMetadata object, "
            "this field should be True if the cleaning has been completed."
        ),
    )
    expect_reply: bool = Field(
        ...,
        description=(
            "Whether you expect a reply from the supervisor based on content of "
            "'reply_msg_to_supervisor'. This is ONLY for receiving replies from the supervisor, "
            "not from worker agents. If you are the/a supervisor (or the router, planner, or "
            "progress reporter), only set this to True if you are expecting a reply from the "
            "main supervisor, NOT from a worker agent. Worker agents will always reply to "
            "'next_agent_prompt' when routed to."
        ),
    )


# ---------------------------------------------------------------------------
# Progress Report
# ---------------------------------------------------------------------------

class ProgressReport(BaseNoExtrasModel):
    """Progress report emitted by agents or supervisor."""

    latest_progress: str = Field(..., description="Latest progress of the agent.")


# ---------------------------------------------------------------------------
# Plan Helper Types & Step Normalization
# ---------------------------------------------------------------------------

Triplet = Tuple[int, str, str]  # (step_number, step_name, step_description)


def _norm(s: Optional[str]) -> str:
    """Normalize whitespace in strings for reliable comparison."""
    return (s or "").strip()


def _triplet_from_raw(d: Dict[str, Any]) -> Triplet:
    """Extract step triplet from raw dictionary."""
    return (
        int(d.get("step_number", 0)),
        _norm(d.get("step_name")),
        _norm(d.get("step_description")),
    )


def _sort_plan_steps(steps: List["PlanStep"]) -> List["PlanStep"]:
    """Ensure plan steps are validated and sorted ascending by step_number."""
    norm = [
        s if isinstance(s, PlanStep) else PlanStep.model_validate(s)
        for s in (steps or [])
    ]
    return sorted(norm, key=lambda s: s.step_number)


def _assert_sorted_completed_no_dups(steps: List["PlanStep"]) -> List["PlanStep"]:
    """Assert completed steps are sorted ascending, completed, and free of duplicate numbers."""
    nums = [s.step_number for s in steps]
    if nums != sorted(nums):
        raise ValueError("completed_steps must be sorted ascending by step_number.")

    for s in steps:
        if s.is_step_complete is not True:
            raise ValueError("All completed_steps must have is_step_complete=True.")

    seen: set[Triplet] = set()
    seen_nums: set[int] = set()
    for s in steps:
        t = (s.step_number, s.step_name, s.step_description)
        if t in seen:
            raise ValueError(f"Duplicate completed step detected: {t}")
        if s.step_number in seen_nums:
            raise ValueError(
                f"Duplicate step_number {s.step_number} in completed_steps"
            )
        seen.add(t)
        seen_nums.add(s.step_number)
    return steps


# ---------------------------------------------------------------------------
# PlanStep Model
# ---------------------------------------------------------------------------

class PlanStep(BaseNoExtrasModel):
    """Single step within an analysis plan."""

    step_number: int = Field(..., description="Step number of the plan.")
    step_name: str = Field(..., description="Name of the step.")
    step_description: str = Field(
        ..., description="Description and detailed instructions for the step."
    )
    is_step_complete: bool = Field(..., description="Whether the step is complete.")
    plan_version: int = Field(..., description="Numeric version of the plan.")


# ---------------------------------------------------------------------------
# Plan Model with Lifecycle Semantics
# ---------------------------------------------------------------------------

class Plan(BaseNoExtrasModel):
    """Analysis plan with explicit lifecycle semantics (Issue #156).

    Lifecycle Invariants:
    1. NEW / LLM-authored plan:
       Normal construction (Plan(...)) or model_validate(data) allocates a fresh,
       unique, globally monotonic version under lock, even if draft input supplied
       a proposed version (e.g. plan_version=1).
    2. RESTORE / persisted plan snapshot:
       Restoration entrypoint (Plan.from_persisted_snapshot(data)) or explicit
       validation context (model_validate(data, context={"restore": True}))
       preserves the stored snapshot version (e.g. 42) without re-allocation.
    3. Allocator High-Water Mark Policy:
       When a trusted plan version is restored, the allocator's high-water mark
       is updated under the same lock so subsequent new plans receive strictly
       greater versions (> restored_version).
    4. Step Synchronization:
       PlanStep.plan_version across plan_steps are synchronized to the parent's
       effective version in both creation and restoration paths.
    """

    plan_version: int = Field(..., description="Numeric version of the plan.")
    plan_title: str = Field(..., description="Title of the plan.")
    plan_summary: str = Field(..., description="Summary of the plan.")
    plan_steps: Annotated[List[PlanStep], AfterValidator(_sort_plan_steps)] = Field(...)

    _lock: ClassVar[threading.Lock] = threading.Lock()
    _current_version: ClassVar[int] = 1
    _ver_assigned: bool = PrivateAttr(default=False)

    @classmethod
    def _allocate_version(cls) -> int:
        """Allocate the next globally monotonic plan version under lock."""
        with cls._lock:
            v = cls._current_version
            cls._current_version += 1
            return v

    @classmethod
    def _update_high_water_mark(cls, version: int) -> None:
        """Update allocator high-water mark so subsequent new plans receive > version."""
        with cls._lock:
            if version >= cls._current_version:
                cls._current_version = version + 1

    @classmethod
    def reset_counter(cls, start: int = 1) -> None:
        """Reset the internal allocator counter to start (used for test isolation)."""
        with cls._lock:
            cls._current_version = start

    @classmethod
    def create_new_plan(cls, **kwargs: Any) -> "Plan":
        """Explicit new-plan factory. Allocates a fresh monotonic plan_version."""
        return cls(**kwargs)

    @classmethod
    def from_persisted_snapshot(
        cls, data: Union[Dict[str, Any], "Plan"]
    ) -> "Plan":
        """Explicit restoration entrypoint for persisted snapshots.

        Preserves the stored plan_version, synchronizes plan steps, and updates
        the allocator high-water mark under lock so subsequently created new plans
        receive strictly increasing versions.
        """
        if isinstance(data, cls):
            cls._update_high_water_mark(data.plan_version)
            return data
        if not isinstance(data, dict):
            raise TypeError(
                f"Snapshot data must be a dict or Plan, got {type(data).__name__}"
            )
        return cls.model_validate(data, context={"restore": True})

    @field_validator("plan_steps", mode="after")
    @classmethod
    def _sync_step_versions_on_assignment(
        cls, steps: List["PlanStep"], info: ValidationInfo
    ) -> List["PlanStep"]:
        pv = info.data.get("plan_version")
        if pv is None:
            return steps
        steps = [
            s if s.plan_version == pv else s.model_copy(update={"plan_version": pv})
            for s in steps
        ]
        nums = [s.step_number for s in steps]
        if any(b <= a for a, b in zip(nums, nums[1:])):
            raise ValueError(
                f"plan_steps must be strictly increasing by step_number, got {nums}"
            )
        return steps

    @model_validator(mode="after")
    def _sync_steps_and_assert_increasing(self, info: ValidationInfo) -> "Plan":
        is_restore = bool(
            info.context
            and (info.context.get("restore") or info.context.get("persisted"))
        )

        if not self._ver_assigned:
            if is_restore:
                # Restoration path: trust and preserve the persisted plan_version
                self._update_high_water_mark(self.plan_version)
                self._ver_assigned = True
            else:
                # New plan path: allocate fresh globally monotonic version
                v = self._allocate_version()
                object.__setattr__(self, "plan_version", v)
                self._ver_assigned = True

        pv = self.plan_version
        self.plan_steps = [
            s if s.plan_version == pv else s.model_copy(update={"plan_version": pv})
            for s in self.plan_steps
        ]

        nums = [s.step_number for s in self.plan_steps]
        if any(b <= a for a, b in zip(nums, nums[1:])):
            raise ValueError(
                f"plan_steps must be strictly increasing by step_number, got {nums}"
            )
        return self


# Backward-compatibility alias for legacy code inspecting Plan._counter
class _PlanCounterWrapper:
    def __iter__(self):
        return self

    def __next__(self) -> int:
        return Plan._allocate_version()


Plan._counter = _PlanCounterWrapper()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CompletedStepsAndTasks Model
# ---------------------------------------------------------------------------

class CompletedStepsAndTasks(BaseNoExtrasModel):
    """Completed steps and tasks tracker."""

    completed_steps: Annotated[
        List[PlanStep], AfterValidator(_assert_sorted_completed_no_dups)
    ] = Field(...)
    finished_tasks: List[str] = Field(
        ...,
        description="List of tasks that have been completed based on the steps of the Plan",
    )
    progress_report: ProgressReport = Field(...)

    @field_validator("completed_steps", mode="before")
    @classmethod
    def _inject_and_dedupe(cls, v, info: ValidationInfo):
        if not isinstance(v, list):
            return v
        plan: Optional[Plan] = (info.context or {}).get("plan")
        pv = plan.plan_version if plan else None

        seen: Dict[Triplet, Dict[str, Any]] = {}
        for item in v:
            d = (
                item.model_dump()
                if isinstance(item, PlanStep)
                else dict(item)
                if hasattr(item, "items") or isinstance(item, dict)
                else {}
            )
            if pv is not None:
                d["plan_version"] = pv
            key = _triplet_from_raw(d)

            prev = seen.get(key)
            cand_score = (
                int(d.get("plan_version", -1)),
                bool(d.get("is_step_complete", False)),
            )
            prev_score = (
                (-1, False)
                if prev is None
                else (
                    int(prev.get("plan_version", -1)),
                    bool(prev.get("is_step_complete", False)),
                )
            )

            if prev is None or cand_score >= prev_score:
                seen[key] = d

        # RC-2 BUG FIX: return dedup_list (sorted ascending) instead of list(seen.values())
        dedup_list = list(seen.values())
        dedup_list.sort(key=lambda d: int(d.get("step_number", 10**9)))
        return dedup_list

    @field_validator("completed_steps", mode="after")
    @classmethod
    def _sorted_no_dups_and_subset(
        cls, steps: List[PlanStep], info: ValidationInfo
    ) -> List[PlanStep]:
        nums = [s.step_number for s in steps]
        if nums != sorted(nums):
            raise ValueError(
                "completed_steps must be sorted ascending by step_number."
            )

        plan: Optional[Plan] = (info.context or {}).get("plan")
        if plan:
            allowed = {
                (ps.step_number, _norm(ps.step_name), _norm(ps.step_description))
                for ps in plan.plan_steps
            }
            for s in steps:
                k = (s.step_number, _norm(s.step_name), _norm(s.step_description))
                if k not in allowed:
                    raise ValueError(
                        f"Completed step {k} is not present in the supplied Plan."
                    )
        return steps


# ---------------------------------------------------------------------------
# Plan Reducer
# ---------------------------------------------------------------------------

def _reduce_plan_keep_sorted(
    a: Optional[Plan], b: Optional[Plan]
) -> Optional[Plan]:
    """LangGraph state reducer merging two Plans by step_number (last-wins)."""
    if a is None:
        return b
    if b is None:
        return a

    steps: List[Any] = []
    if a.plan_steps:
        steps.extend(a.plan_steps)
    if b.plan_steps:
        steps.extend(b.plan_steps)

    norm = [
        s if isinstance(s, PlanStep) else PlanStep.model_validate(s)
        for s in steps
    ]
    by_num = {s.step_number: s for s in norm}
    merged_sorted_steps = [by_num[k] for k in sorted(by_num)]

    merged = {
        **a.model_dump(),
        **b.model_dump(),
        "plan_steps": merged_sorted_steps,
    }
    return Plan.model_validate(merged)
