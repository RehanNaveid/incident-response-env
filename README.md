---
title: IncidentIQ
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
python_version: "3.12"
app_file: app.py
pinned: false
---

# IncidentIQ — SRE Incident Response Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com)

## Description and Motivation

Site Reliability Engineers respond to production incidents under time pressure, working from incomplete information — logs, metrics, and service dependencies — to diagnose root causes and apply targeted fixes before SLA windows expire. This is a structured reasoning task that real engineers perform daily, and one where AI agents are increasingly expected to assist.

**IncidentIQ** simulates this task as a reinforcement learning environment. An agent receives live system telemetry, must reason about root causes from log evidence, and execute a correct sequence of actions to resolve the incident. The environment models real failure patterns:

- Connection pool exhaustion
- Cascading config deploys
- Memory leaks
- Certificate expiry
- Upstream rate limiting
- Database degradation

Training agents on this environment develops capabilities in structured diagnostic reasoning, evidence-based decision making, and sequential planning under constraints — skills that transfer directly to real SRE automation.

---

## Action Space

Actions are **free-text commands**. The environment parses intent from the text.

| Format | Description |
|---|---|
| `investigate <service>` | Examine a specific service. Service name must match **AFFECTED SERVICES** exactly. |
| `assign to <team>` | Assign the incident to a team. Team name must match **TEAM ROSTER** exactly. |
| `mitigate: <fix>` | Apply a targeted fix. The fix keyword must appear in the `ERROR` or `CRIT` logs. |
| `escalate` | Escalate the incident severity. |
| `resolve` | Close the incident. Only valid after a confirmed mitigation. |

**Enforcement rules:**
- Mitigation requires prior investigation of at least one affected service.
- Resolve requires a confirmed mitigation.
- Repeated identical actions incur penalties.

---

## Observation Space

Each `step()` call returns an `IncidentObservation` object with these fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | `string` | Task identifier |
| `incident_description` | `string` | Human-readable incident title |
| `affected_services` | `list[string]` | Services involved in the incident |
| `severity` | `P0 \| P1 \| P2` | Incident severity level |
| `logs` | `list[string]` | Last 8 log lines with ISO 8601 timestamps |
| `metrics` | `list[object]` | Per-service `error_rate_pct`, `latency_p99_ms`, `throughput_rps`, `status` |
| `feedback` | `string` | Natural language result of the last action |
| `reward` | `float` | Step reward for the last action |
| `score` | `float [0,1]` | Cumulative normalized episode progress |
| `sla_remaining` | `int` | Minutes remaining before SLA breach |
| `team_roster` | `dict` | `team_name → available \| busy` |

---

## Tasks

### Task 1 — Single Service Outage *(Easy, max 10 steps)*

One service is down with a clear root cause visible in the logs. The agent must investigate the affected service, assign the correct team, apply the matching mitigation keyword from the error log, and resolve. A capable agent should complete this in 4–5 steps.

### Task 2 — Cascading Failure *(Medium, max 18 steps)*

Three services fail in sequence after a config deploy. The agent must investigate all three, identify the root cause service, apply the correct rollback, and resolve before a tight SLA expires. Requires systematic investigation before mitigation.

### Task 3 — Ambiguous Payment Degradation *(Hard, max 25 steps)*

The payment service degrades with three plausible root causes and two deliberate red herrings in the logs. The agent must investigate multiple hypothesis domains (upstream rate limiting, database issues, resource exhaustion), identify the real cause, and apply the correct mitigation. The optional `reasoning` field in actions is evaluated for partial credit, making this the only task that rewards chain-of-thought, not just actions.

---

## Reward Design

Rewards are **dense and continuous** across the full episode:

| Action | Reward |
|---|---|
| Investigate correct service | `+0.15` to `+0.23` |
| Assign correct team | `+0.10` |
| Apply correct mitigation keywords | `+0.20` to `+0.52` |
| Mitigate without investigating first | `−0.25` |
| Resolve before mitigation confirmed | `−0.40` |
| Repeated action | `−0.20` |
| Unknown action | `−0.20` |
| SLA breach (past 60% of window) | Graduated decay |

Final episode score is computed by a **deterministic grader** applied to the full action history. Graders check investigation completeness, team assignment accuracy, mitigation keyword match, resolution, and efficiency. All scores are in `[0.0, 1.0]`.

---

## Baseline Results

Run with `seed=42`, model `openai/gpt-4o-mini`, no task-specific hints in prompt.

| Task | Difficulty | Steps Used | Score |
|---|---|---|---|
| `single_service_outage` | Easy | 7 | **0.85** |
| `cascading_failure` | Medium | 7 | **0.96** |
| `ambiguous_payment_degradation` | Hard | 10 | **0.99** |
| **Average** | | **8.0** | **0.93** |

<details>
<summary>Full inference log</summary>

```
[START] task=single_service_outage env=incident_response_env model=openai/gpt-4o-mini
[STEP] step=1 action=investigate payment-service reward=0.25 done=false error=null
[STEP] step=2 action=assign to payments-oncall reward=0.10 done=false error=null
[STEP] step=3 action=mitigate: increase memory limit for payment-service pods reward=0.32 done=false error=null
[STEP] step=4 action=mitigate: increase memory limit for payment-service pods reward=0.40 done=false error=null
[STEP] step=5 action=mitigate: increase memory limit for payment-service pods reward=-0.26 done=false error=null
[STEP] step=6 action=mitigate: increase memory limit for payment-service pods reward=-0.39 done=false error=null
[STEP] step=7 action=resolve reward=0.30 done=true error=null
[END] success=true steps=7 score=0.85 rewards=0.25,0.10,0.32,0.40,-0.26,-0.39,0.30

[START] task=cascading_failure env=incident_response_env model=openai/gpt-4o-mini
[STEP] step=1 action=investigate auth-service reward=0.25 done=false error=null
[STEP] step=2 action=investigate api-gateway reward=0.22 done=false error=null
[STEP] step=3 action=investigate user-service reward=0.19 done=false error=null
[STEP] step=4 action=mitigate: rollback config token_verifier_v2 in auth-service reward=0.42 done=false error=null
[STEP] step=5 action=mitigate: rollback config token_verifier_v2 in auth-service reward=0.42 done=false error=null
[STEP] step=6 action=mitigate: rollback config token_verifier_v2 in auth-service reward=-0.26 done=false error=null
[STEP] step=7 action=resolve reward=0.28 done=true error=null
[END] success=true steps=7 score=0.96 rewards=0.25,0.22,0.19,0.42,0.42,-0.26,0.28

[START] task=ambiguous_payment_degradation env=incident_response_env model=openai/gpt-4o-mini
[STEP] step=1 action=investigate payment-service reward=-0.15 done=false error=null
[STEP] step=2 action=investigate payment-service db connection reward=-0.68 done=false error=null
[STEP] step=3 action=investigate payment-service rate limit reward=-0.46 done=false error=null
[STEP] step=4 action=investigate payment-service memory heap reward=-0.34 done=false error=null
[STEP] step=5 action=assign to payments-oncall reward=0.10 done=false error=null
[STEP] step=6 action=mitigate: increase memory allocation for payment-service reward=0.52 done=false error=null
[STEP] step=7 action=mitigate: increase memory allocation for payment-service reward=0.61 done=false error=null
[STEP] step=8 action=mitigate: increase memory allocation for payment-service reward=-0.07 done=false error=null
[STEP] step=9 action=mitigate: increase memory allocation for payment-service reward=-0.40 done=false error=null
[STEP] step=10 action=resolve reward=0.30 done=true error=null
[END] success=true steps=10 score=0.99 rewards=-0.15,-0.68,-0.46,-0.34,0.10,0.52,0.61,-0.07,-0.40,0.30

[SUMMARY] tasks=3 avg_score=0.9308 total_steps=24 all_success=True
  single_service_outage               [████████████████░░░░] 0.8460 (7 steps)
  cascading_failure                   [███████████████████░] 0.9587 (7 steps)
  ambiguous_payment_degradation       [███████████████████░] 0.9878 (10 steps)
```

</details>

---

## Setup and Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
export API_BASE_URL="https://your-llm-endpoint/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-api-key"
# For local testing
export ENV_URL="http://localhost:7860"

# For Hugging Face deployment (used by evaluator)
export ENV_URL="https://<your-space-name>.hf.space" # default; only change if server runs elsewhere
```

### Running

```bash
# Terminal 1 — start the environment server
python app.py

# Terminal 2 — run the inference script
python inference.py
```

### Running a Single Task

```bash
TASK_IDS_OVERRIDE=single_service_outage python inference.py
```

### Docker

```bash
docker build -t incidentiq .
docker run -p 7860:7860 \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  -e HF_TOKEN=$HF_TOKEN \
  incidentiq
```

### Verifying the Environment

```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "single_service_outage", "seed": 42}' \
  | python -m json.tool
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode. Body: `{"task_id": "...", "seed": 42}` |
| `/step` | POST | Send one action. Body: `{"action": {"action": "..."}}` |
| `/state` | GET | Get current episode state including ground truth for grading |
| `/incident-meta` | GET | Get incident metadata used by graders |
| `/runbook` | GET | Get diagnostic hints for an affected service |
| `/tasks` | GET | List all available tasks |
| `/health` | GET | Server health check |

---

## OpenEnv Compliance

- Full `step()` / `reset()` / `state()` implementation
- Typed Pydantic models for all actions and observations
- Deterministic graders with no LLM calls
- Reproducible with fixed seed
- Dense reward across full trajectory
- Session isolation for parallel evaluation runs

---

## Project Structure

incident-response-env/
├── server/                         # Core environment server (FastAPI + OpenEnv)
│   ├── __init__.py
│   ├── app.py                      # FastAPI app exposing /reset, /step, /state, /health
│   ├── environment.py              # Main environment logic (step/reset/state, reward)
│   ├── incidents.py                # Deterministic incident generator (seed-based)
│   ├── simulator.py                # Dynamic simulation (logs + metrics evolution)
│   ├── tasks.py                    # Task configs (difficulty, max_steps, rewards)
│
├── models.py                       # Pydantic models (Action, Observation, State)
├── inference.py                    # Baseline agent (OpenAI-compatible client)
├── client.py                       # Optional client helper for interacting with env
│
├── openenv.yaml                    # OpenEnv metadata (tasks, spaces, entrypoint)
├── Dockerfile                      # Container setup for HF Spaces deployment
├── pyproject.toml                  # Project config (used by uv)
├── uv.lock                         # Dependency lock file (reproducible builds)
├── requirements.txt                # Python dependencies (fallback install)
│
├── .env                            # Local environment variables (not committed)
├── .env.example                    # Template for required env variables
├── .gitignore                      # Ignore rules
│
├── validate-submission.sh          # Pre-submission validation script
├── README.md                       # Project documentation
│
├── venv/ or .venv/                 # Virtual environment (local only, ignored)
└── __pycache__/                    # Python cache (auto-generated)

---
The inference script uses an OpenAI-compatible client interface configured via API_BASE_URL and MODEL_NAME.
## License

MIT