"""
Pydantic models for the Incident Response OpenEnv environment.

IMPORTANT: IncidentAction extends Action, IncidentObservation extends
Observation, and IncidentState extends State from the OpenEnv framework.
This ensures the serializer can access required fields (reward, done,
metadata) on the observation object without AttributeError.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from openenv.core.env_server.types import Action, Observation, State


VALID_TASK_IDS = {
    "single_service_outage",
    "cascading_failure",
    "ambiguous_payment_degradation",
}


# ---------------------------------------------------------------------------
# Action  (extends framework Action — inherits metadata field)
# ---------------------------------------------------------------------------

class IncidentAction(Action):
    """Agent action for one step.

    action  — free-form command string; environment parses intent from it.
    reasoning — optional scratchpad; used for partial credit in task 3 grader.
    """
    action: str = Field(..., min_length=1, max_length=512)
    reasoning: Optional[str] = Field(None, max_length=1024)


# ---------------------------------------------------------------------------
# Observation  (extends framework Observation — inherits done, reward, metadata)
# ---------------------------------------------------------------------------

class ServiceMetrics(BaseModel):
    service: str
    error_rate_pct: float = Field(..., ge=0.0, le=100.0)
    latency_p99_ms: int   = Field(..., ge=0)
    throughput_rps: int   = Field(..., ge=0)
    status: Literal["ok", "degraded", "down"]


class IncidentObservation(Observation):
    """What the agent sees each step.

    Inherits from openenv Observation which provides:
        - done: bool (default False)
        - reward: float | None (default None)
        - metadata: Dict[str, Any] (default {})

    The framework serializer (serialize_observation) reads these three
    fields directly. They MUST exist on the observation object.

    Deliberately excludes: valid_mitigations, root_cause, correct_team.
    Agent must reason from incident_description + logs + metrics.
    """
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Domain-specific observation fields
    task_id: str = ""
    incident_description: str = ""
    affected_services: List[str] = Field(default_factory=list)
    severity: Literal["P0", "P1", "P2"] = "P2"
    logs: List[str] = Field(default_factory=list)
    metrics: List[ServiceMetrics] = Field(default_factory=list)
    feedback: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    step_count: int = Field(default=0, ge=0)
    resolved: bool = False
    last_action: Optional[str] = None
    outcome: Dict[str, Any] = Field(default_factory=dict)
    sla_remaining: int = 0
    team_roster: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal state  (extends framework State — inherits episode_id, step_count)
# ---------------------------------------------------------------------------

class IncidentState(State):
    """Internal state of the incident-response environment."""
    task_id: str = ""
    cumulative_reward: float = 0.0
    resolved: bool = False
    seed: int = 0
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: IncidentObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "single_service_outage"
    seed: Optional[int] = None
    episode_id: Optional[str] = None

    @field_validator("task_id")
    @classmethod
    def check_task_id(cls, v: str) -> str:
        if v not in VALID_TASK_IDS:
            raise ValueError(f"task_id must be one of {sorted(VALID_TASK_IDS)}, got {v!r}")
        return v


class StepRequest(BaseModel):
    action: IncidentAction