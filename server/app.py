"""
FastAPI application for the Incident Response Environment.

Uses OpenEnv's ``create_fastapi_app`` to expose the environment over HTTP.

Enhancements over baseline:
    - ``_BoundedSessionStore``: LRU-evicting dict that prevents OOM from
      abandoned sessions (max 100 concurrent).
    - ``/runbook`` endpoint: authentic SRE runbook lookup that enriches the
      observation space without giving away the answer.
    - ``/session`` DELETE: explicit session cleanup.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

from openenv.core.env_server import create_fastapi_app
from fastapi import Header
from fastapi.responses import JSONResponse

from models import IncidentAction, IncidentObservation
from server.environment import IncidentResponseEnv
from server.tasks import TASK_CONFIGS


# ===================================================================
# Bounded session store — prevents memory leaks from abandoned sessions
# ===================================================================

_MAX_SESSIONS: int = 100


class _BoundedSessionStore:
    """Thread-safe, bounded, LRU-evicting session store.

    When ``max_size`` is exceeded, the least-recently-used session is
    evicted.  All public methods are protected by a threading lock.
    """

    def __init__(self, max_size: int = _MAX_SESSIONS) -> None:
        self._store: OrderedDict[str, IncidentResponseEnv] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[IncidentResponseEnv]:
        with self._lock:
            if session_id in self._store:
                self._store.move_to_end(session_id)
                return self._store[session_id]
            return None

    def set(self, session_id: str, env: IncidentResponseEnv) -> None:
        with self._lock:
            if session_id in self._store:
                self._store.move_to_end(session_id)
            self._store[session_id] = env
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Global session store instance
_sessions = _BoundedSessionStore()

# Default session for single-user / local development
_DEFAULT_SESSION = "default"


# ===================================================================
# App factory
# ===================================================================

# Instantiate one env for the /incident-meta convenience endpoint
_env = IncidentResponseEnv()

# Pass the class + type hints — framework instantiates per session.
app = create_fastapi_app(lambda: _env, IncidentAction, IncidentObservation)


# ===================================================================
# Custom endpoints
# ===================================================================
@app.post("/reset")
def reset_override(
    body: ResetRequest,
    x_session_id: str = Header(default=_DEFAULT_SESSION),
):
    sid = x_session_id or _DEFAULT_SESSION
    env = _sessions.get(sid)
    if env is None:
        env = IncidentResponseEnv()
        _sessions.set(sid, env)
    obs = env.reset(
        task_id=body.task_id,
        seed=body.seed,
        episode_id=body.episode_id,
    )
    return {"observation": obs.model_dump(), "done": False, "reward": 0.0, "info": {}}


@app.post("/step")
def step_override(
    body: StepRequest,
    x_session_id: str = Header(default=_DEFAULT_SESSION),
):
    sid = x_session_id or _DEFAULT_SESSION
    env = _sessions.get(sid)
    if env is None:
        return JSONResponse(status_code=400, content={"error": "Call /reset first."})
    obs = env.step(body.action)
    return {
        "observation": obs.model_dump(),
        "done": obs.done,
        "reward": obs.reward,
        "info": obs.metadata,
    }


@app.get("/state")
def state_override(x_session_id: str = Header(default=_DEFAULT_SESSION)):
    sid = x_session_id or _DEFAULT_SESSION
    env = _sessions.get(sid) or _env
    s = env.state()
    return s.model_dump()

@app.get("/tasks")
def list_tasks():
    """Enumerate available tasks — required by ``openenv validate``."""
    return JSONResponse({
        "tasks": [
            {
                "id": task_id,
                "max_steps": cfg["max_steps"],
                "max_reward": cfg["max_reward"],
                "difficulty": cfg.get("difficulty", ""),
                "description": cfg.get("description", ""),
            }
            for task_id, cfg in TASK_CONFIGS.items()
        ]
    })


@app.get("/health")
def health():
    """Health check endpoint for Docker HEALTHCHECK and validator."""
    return {
        "status": "ok",
        "env": "incident_response_env",
        "active_sessions": len(_sessions),
    }


@app.get("/incident-meta")
def incident_meta(
    x_session_id: str = Header(default=_DEFAULT_SESSION),
):
    env = _sessions.get(x_session_id) or _env
    inc = env.incident_data
    if not inc:
        return {
            "root_cause": "",
            "team": "",
            "correct_team": "",
            "valid_mitigations": [],
            "root_cause_service": "",
            "affected_services": [],
        }
    return {
        "root_cause":        inc.get("root_cause", ""),
        "team":              inc.get("team", ""),
        "correct_team":      inc.get("correct_team", ""),
        "valid_mitigations": inc.get("valid_mitigations", []),
        "root_cause_service": inc.get("root_cause_service", ""),
        "affected_services": inc.get("affected_services", []),
        "runbook_queries":   env._runbook_queries,
    }


# -------------------------------------------------------------------
# /runbook — authentic SRE runbook endpoint
# -------------------------------------------------------------------

@app.get("/runbook")
def runbook(
    service: str,
    x_session_id: str = Header(default=_DEFAULT_SESSION),
) -> Dict[str, Any]:
    """Return a runbook entry for *service*.

    Provides diagnostic steps, common root causes, and escalation paths
    without giving away the incident's actual root cause.  This enriches
    the observation space and rewards agents that use available tooling.
    """
    inc = _env.incident_data
    team_map: Dict[str, str] = inc.get("team_map", {})

    if service not in inc.get("affected_services", []):
        return {
            "service": service,
            "found": False,
            "message": "No runbook entry — service not in incident scope.",
        }

    if service not in _env._runbook_queries:
        _env._runbook_queries.append(service)

    if inc.get("root_cause") == "db_overload" and inc.get("affected_services") == ["payment-service"]:
        return {
            "service": service,
            "found": True,
            "title": "Runbook: Payment Latency",
            "owning_team": team_map.get(service, "unknown"),
            "diagnostic_steps": [
                "Restart payment-service",
                "Scale replicas if needed",
                "Check DB connections if issue persists",
            ],
            "escalation_path": team_map.get(service, "platform"),
        }

    return {
        "service": service,
        "found": True,
        "owning_team": team_map.get(service, "unknown"),
        "diagnostic_steps": [
            f"kubectl logs -l app={service} --tail=50",
            f"kubectl describe pod -l app={service}",
            "Check upstream dependency health in Grafana",
            "Review recent deploys in the deploy log",
        ],
        "common_root_causes": [
            "Connection pool exhaustion",
            "Config change after deploy",
            "Memory leak or OOM",
            "Upstream rate limiting",
        ],
        "escalation_path": team_map.get(service, "platform"),
    }


# -------------------------------------------------------------------
# /session — explicit session cleanup
# -------------------------------------------------------------------

@app.delete("/session")
def delete_session(
    x_session_id: str = Header(default=_DEFAULT_SESSION),
) -> Dict[str, Any]:
    """Explicitly delete a session to free memory."""
    deleted = _sessions.delete(x_session_id)
    return {"deleted": deleted, "session_id": x_session_id}


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    """Entry point for direct execution."""
    import os
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
