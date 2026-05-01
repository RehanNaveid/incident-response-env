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

from models import IncidentAction, IncidentObservation, ResetRequest, StepRequest
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

# Remove any /state route the framework registered so our custom override wins.
# The framework passes the bound method object instead of calling it, producing
# a ResponseValidationError 500.  Our @app.get("/state") below calls env.state()
# correctly and returns s.model_dump().
app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/state"]


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

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "incident-response-env",
        "health": "/health",
        "docs": "/docs"
    }

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
    dag = inc.get("fan_in_dag", {})
    return {
        "root_cause":         inc.get("root_cause", ""),
        "team":               inc.get("team", ""),
        "correct_team":       inc.get("correct_team", ""),
        "valid_mitigations":  inc.get("valid_mitigations", []),
        "root_cause_service": inc.get("root_cause_service", ""),
        "affected_services":  inc.get("affected_services", []),
        "runbook_queries":    env._runbook_queries,
        # Causal fan-in DAG — agent-visible topology (nodes + observable edges only)
        "fan_in_dag": {
            "nodes": dag.get("nodes", []),
            "edges": dag.get("edges", []),
        } if dag else {},
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
    # Use the per-session env — during multi-session training (6 concurrent
    # rollouts) the global _env holds a different incident than the caller's.
    env = _sessions.get(x_session_id) or _env
    inc = env.incident_data
    team_map: Dict[str, str] = inc.get("team_map", {})

    if service not in inc.get("affected_services", []):
        return {
            "service": service,
            "found": False,
            "message": "No runbook entry — service not in incident scope.",
        }

    if service not in env._runbook_queries:
        env._runbook_queries.append(service)

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


# ===================================================================
# Gradio UI  (Option C — UI button demo)
# ===================================================================

def _build_gradio_ui():
    """Build and return the Gradio Blocks interface.

    Mounted at /ui so the FastAPI REST endpoints remain available
    at their existing paths (/reset, /step, /state …).

    The agent loop calls inference.run_task() directly (USE_LOCAL_ENV mode)
    and yields one line at a time so the textbox updates live.
    """
    try:
        import gradio as gr
    except ImportError:
        return None   # Gradio not installed — skip UI gracefully

    import sys, io, threading, queue as _queue

    TASK_CHOICES = [
        ("🔴 Single service outage",         "single_service_outage"),
        ("🟠 Cascading failure",              "cascading_failure"),
        ("🟡 Ambiguous payment degradation",  "ambiguous_payment_degradation"),
    ]

    def _run_agent(task_label: str, seed: int) -> str:
        """Run one episode and return the full transcript as a string."""
        # Map display label → task_id
        task_id = next(
            (tid for lbl, tid in TASK_CHOICES if lbl == task_label),
            "single_service_outage",
        )

        # Capture stdout (log_start / log_step / log_end / [GRADE] lines)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        try:
            import inference as _inf
            # Force local-env mode for the UI — no HTTP server required
            _inf.USE_LOCAL_ENV = True
            result = _inf.run_task(task_id=task_id, seed=int(seed))
        except Exception as exc:
            import traceback
            buf.write(f"\n[ERROR] {exc}\n")
            traceback.print_exc(file=buf)
            result = {}
        finally:
            sys.stdout = old_stdout

        transcript = buf.getvalue()

        # Append a visual summary footer
        score   = result.get("score",   0.0)
        steps   = result.get("steps",   0)
        success = result.get("success", False)
        status  = "✅ RESOLVED" if success else "❌ NOT RESOLVED"
        footer = (
            f"\n{'─' * 60}\n"
            f"  {status}   score={score:.3f}   steps={steps}\n"
            f"{'─' * 60}"
        )
        return transcript + footer

    with gr.Blocks(title="IncidentIQ — RL Agent Demo") as demo:
        gr.Markdown(
            """
            # 🚨 IncidentIQ — Incident Response RL Agent
            Select a scenario and click **Run Agent** to watch the trained LoRA model
            investigate, mitigate, and resolve the incident in real time.
            """
        )

        with gr.Row():
            task_dd = gr.Dropdown(
                choices=[lbl for lbl, _ in TASK_CHOICES],
                value=TASK_CHOICES[0][0],
                label="Incident Scenario",
            )
            seed_nb = gr.Number(value=42, label="Seed", precision=0, minimum=0)

        run_btn  = gr.Button("▶ Run Agent", variant="primary")
        clear_btn = gr.Button("🗑 Clear", variant="secondary")

        output_box = gr.Textbox(
            label="Agent Transcript",
            lines=30,
            max_lines=60,
            interactive=False,
        )

        run_btn.click(
            fn=_run_agent,
            inputs=[task_dd, seed_nb],
            outputs=output_box,
        )
        clear_btn.click(fn=lambda: "", outputs=output_box)

    return demo


# Mount Gradio at /ui  (coexists with the FastAPI REST API)
_gradio_ui = _build_gradio_ui()
if _gradio_ui is not None:
    try:
        import gradio as gr
        app = gr.mount_gradio_app(app, _gradio_ui, path="/ui")
    except Exception as _e:
        print(f"[WARN] Gradio UI mount failed: {_e}", flush=True)


if __name__ == "__main__":
    main()
