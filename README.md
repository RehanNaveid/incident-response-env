# IncidentIQ — Causal Inference for Distributed System Failures

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/RehanNaveid/incident-response-env)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com)
[![Model](https://img.shields.io/badge/LoRA-rehannaveid%2Fincidentiq--lora-orange)](https://huggingface.co/rehannaveid/incidentiq-lora)

> An RL-trained LLM agent that identifies true root causes in multi-service outages through sequential hypothesis testing under uncertainty — not keyword matching. The agent builds its own belief distribution from raw, noisy signals across a deceptive causal graph.

**POMDP world modeling · GRPO + Unsloth · Causal DAG · Multi-reward RL**
---

## Live Links

- Live Environment Demo:
  https://huggingface.co/spaces/RehanNaveid/incident-response-env

- Trained LoRA Model:
  https://huggingface.co/RehanNaveid/incidentiq-lora


---

## Why RL? The gap that makes it necessary

A prompted GPT-4o-mini solves the old version (visible root cause in logs) with a 0.93 score — zero RL needed. The new version hides the root cause inside a latent causal DAG with deceptive signals. No static prompt can solve a POMDP that changes every episode.

| Version | Setup | Baseline score | RL needed? |
|---|---|---|---|
| Old | Root cause visible in logs — agent reads error → keyword match → fix | **0.93** | No |
| New | Root cause hidden in latent causal DAG — agent infers structure from ambiguous logs | **~0.25** | Yes |

The gap from 0.25 to 0.85+ is exactly what GRPO trains into.

---

## What makes this environment hard

Four properties together make reasoning unavoidable — not just "more nodes":

**Fan-in ambiguity** — `api-gateway` failure has 3 possible upstream causes: `auth-service`, `db-service`, `cache-service`. Root is not obvious from any single observation. Agent must disambiguate.

**Red herrings** — `user-service` logs show errors but is perfectly healthy. Observation ≠ truth. Agent must learn to distrust surface-level signals.

**Temporal delay** — `db-service` fails at t=0, api fails at t=2, user fails at t=3. Symptoms appear disconnected from cause.

**Observation noise** — Fake latency spikes, intermittent failures, misleading log lines. Agent must filter signal from noise across multiple investigation steps.

---

## Reward design

Rewards are **dense and continuous** across the full episode, with four independent graders:

| Reward | Weight | What it measures |
|---|---|---|
| R1 — root cause accuracy | 0.60 | Did the agent mitigate the actual root node? Penalty if symptom node mitigated. |
| R2 — belief calibration | 0.40 | Rewards inference trajectory quality. Root probability must increase over steps. Final belief must rank root highest. |

`final_score = R1 × 0.60 + R2 × 0.40`

| Action | Reward |
|---|---|
| Investigate correct service | `+0.15` to `+0.23` |
| Assign correct team | `+0.10` |
| Apply correct mitigation keywords | `+0.20` to `+0.52` |
| Mitigate without investigating first | `−0.25` |
| Resolve before mitigation confirmed | `−0.40` |
| Repeated identical action | `−0.20` |

---

## Trained model results

Model: `rehannaveid/incidentiq-lora` (Qwen2.5-7B-Instruct + LoRA, trained with GRPO via Unsloth), `seed=42`.

| Task | Difficulty | Steps | R1 | R2 | Score | Success |
|---|---|---|---|---|---|---|
| `single_service_outage` | Easy | 11 | 0.90 | 0.67 | **0.81** | ✅ |
| `cascading_failure` | Medium | 18 | 0.41 | 0.62 | **0.49** | ✅ |
| `ambiguous_payment_degradation` | Hard | 9 | 0.70 | 0.27 | **0.53** | ✅ |
| **Average** | | **12.7** | **0.67** | **0.52** | **0.61** | **3/3** |


![image](https://cdn-uploads.huggingface.co/production/uploads/69d5e7a8bd2bdf61c0d49700/bAEH80ZiS8HVCmDjYTmG4.png)


![image](https://cdn-uploads.huggingface.co/production/uploads/69d5e7a8bd2bdf61c0d49700/KhSTgZ4RYaWPalnqwGkGn.png)

<details>
<summary>Full inference log</summary>

```
[START] task=single_service_outage env=incident_response_env model=rehannaveid/incidentiq-lora
[STEP] step=1  action=investigate payment-service              reward=0.48  done=false error=null
[STEP] step=2  action=assign to dev-team                       reward=0.25  done=false error=null
[STEP] step=3  action=mitigate: restart service                reward=0.84  done=false error=null
[STEP] step=4  action=mitigate: scale_up                       reward=0.24  done=false error=null
[STEP] step=5  action=investigate stripe-api                   reward=0.07  done=false error=null
[STEP] step=6  action=resolve                                  reward=-0.35 done=false error=null
[STEP] step=7  action=investigate: payment-service             reward=-0.22 done=false error=null
[STEP] step=8  action=assign to backend-team                   reward=0.16  done=false error=null
[STEP] step=9  action=resolve                                  reward=-0.53 done=false error=null
[STEP] step=10 action=resolve                                  reward=-0.49 done=false error=null
[STEP] step=11 action=resolve                                  reward=-0.54 done=true  error=null
[GRADE] task=single_service_outage R1=0.9000 R2=0.6742 score=0.8097
[END] success=true steps=11 score=0.81

[START] task=cascading_failure env=incident_response_env model=rehannaveid/incidentiq-lora
[STEP] step=1  action=investigate cache-cluster                reward=0.41  done=false error=null
[STEP] step=2  action=investigate auth-service                 reward=0.48  done=false error=null
[STEP] step=3  action=assign to database                       reward=0.29  done=false error=null
[STEP] step=4  action=investigate api-gateway                  reward=0.34  done=false error=null
[STEP] step=5  action=mitigate: restart payment-service db     reward=0.21  done=false error=null
[STEP] step=6  action=resolve                                  reward=-0.04 done=false error=null
[STEP] step=7  action=investigate user-service                 reward=0.09  done=false error=null
[STEP] step=8-18 action=resolve (repeated)                     reward<0     done=true  error=null
[GRADE] task=cascading_failure R1=0.4100 R2=0.6180 score=0.4932
[END] success=true steps=18 score=0.49

[START] task=ambiguous_payment_degradation env=incident_response_env model=rehannaveid/incidentiq-lora
[STEP] step=1  action=investigate payment-service              reward=0.47  done=false error=null
[STEP] step=2  action=assign to database                       reward=0.23  done=false error=null
[STEP] step=3  action=investigate: memory_leak                 reward=0.17  done=false error=null
[STEP] step=4  action=investigate rate_limit                   reward=0.04  done=false error=null
[STEP] step=5  action=assign to backend-team                   reward=-0.15 done=false error=null
[STEP] step=6  action=mitigate: restart service                reward=0.41  done=false error=null
[STEP] step=7  action=resolve                                  reward=-0.35 done=false error=null
[STEP] step=8  action=resolve                                  reward=-0.95 done=false error=null
[STEP] step=9  action=resolve                                  reward=-0.86 done=true  error=null
[GRADE] task=ambiguous_payment_degradation R1=0.7000 R2=0.2661 score=0.5264
[END] success=true steps=9 score=0.53
```

</details>

---

## GPT-4o-mini baseline (old environment)

Run with `seed=42`, model `openai/gpt-4o-mini`, original environment (root cause visible in logs).

| Task | Steps | Score |
|---|---|---|
| `single_service_outage` | 7 | **0.85** |
| `cascading_failure` | 7 | **0.96** |
| `ambiguous_payment_degradation` | 10 | **0.99** |
| **Average** | **8.0** | **0.93** |

This score is why RL is justified — the old environment was already solved. The new deceptive environment drops the untrained baseline to ~0.25.

---

## Belief architecture

The critical design choice: belief must be **constructed by the agent**, not pre-computed by the environment.

**Wrong — privileged state:**
```json
{ "logs": [...], "belief": {"auth": 0.6, "api": 0.3} }
```
Environment precomputes the hard part. R2 becomes tautological. Not a real POMDP.

**Correct — latent cognition:**
```json
{ "logs": [...], "metrics": [...], "prev_actions": [...] }
```
Agent outputs `Thought → Belief → Action`. Belief is the agent's learned inference. R2 rewards quality of the inference trajectory.

Required output format (for GRPO parseability):
```json
{
  "thought": "auth latency increasing → possible upstream issue",
  "belief": {"auth": 0.7, "api": 0.2, "db": 0.1},
  "action": "investigate auth-service"
}
```

Parse failures assign R2=0 but keep the rollout in the batch — R1/R3/R4 still contribute.

---

## GRPO training loop

1. **Observation** — agent sees raw logs, metrics, timestamps, previous actions. No belief. No hints about root cause.
2. **6–8 rollouts** — same prompt, different action sequences. Variance is essential for GRPO.
3. **Score each** — R1–R2 grade every trajectory. Root hit = high reward. Symptom fix = penalty.
4. **Gradient update** — TRL pushes weights toward above-average rollouts. Unsloth: 2× faster, 60% less memory via 4-bit QLoRA.

---

## Tasks

### Task 1 — Single Service Outage *(Easy, max 12 steps)*

One service is down with a clear root cause visible in the logs. The agent must investigate the affected service, assign the correct team, apply the matching mitigation keyword from the error log, and resolve. A capable agent should complete this in 4–5 steps.

### Task 2 — Cascading Failure *(Medium, max 18 steps)*

Three services fail in sequence after a config deploy. The agent must investigate all three, identify the root cause service, apply the correct rollback, and resolve before a tight SLA expires. Requires systematic investigation before mitigation.

### Task 3 — Ambiguous Payment Degradation *(Hard, max 25 steps)*

The payment service degrades with three plausible root causes and two deliberate red herrings in the logs. The agent must investigate multiple hypothesis domains (upstream rate limiting, database issues, resource exhaustion), identify the real cause, and apply the correct mitigation. The `reasoning` field in actions is evaluated for partial credit — the only task that rewards chain-of-thought, not just actions.

---

## Action space

| Format | Description |
|---|---|
| `investigate <service>` | Examine a specific service. Must match **AFFECTED SERVICES** exactly. |
| `assign to <team>` | Assign the incident to a team. Must match **TEAM ROSTER** exactly. |
| `mitigate: <fix>` | Apply a targeted fix. The fix keyword must appear in `ERROR` or `CRIT` logs. |
| `escalate` | Escalate the incident severity. |
| `resolve` | Close the incident. Only valid after a confirmed mitigation. |

Enforcement rules: mitigation requires prior investigation; resolve requires confirmed mitigation; repeated identical actions incur penalties.

---

## Observation space

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

## Setup and usage

### Requirements

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### Environment variables

```bash
export API_BASE_URL="https://your-llm-endpoint/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:7860"   # only needed for HTTP mode
```

### Run with the trained LoRA model (requires GPU)

```bash
USE_TRAINED_MODEL=1 MODEL_ID="rehannaveid/incidentiq-lora" python inference.py
```

### Run with an OpenAI-compatible API (no GPU needed)

```bash
USE_TRAINED_MODEL=0 \
API_BASE_URL="https://openrouter.ai/api/v1" \
MODEL_NAME="qwen/qwen2.5-7b-instruct" \
HF_TOKEN="your-key" \
python inference.py
```

### Run a single task

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

### Verify the environment

```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "single_service_outage", "seed": 42}' \
  | python -m json.tool
```

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode. Body: `{"task_id": "...", "seed": 42}` |
| `/step` | POST | Send one action. Body: `{"action": {"action": "..."}}` |
| `/state` | GET | Current episode state including ground truth for grading |
| `/incident-meta` | GET | Incident metadata used by graders |
| `/runbook` | GET | Diagnostic hints for an affected service |
| `/tasks` | GET | List all available tasks |
| `/health` | GET | Server health check |

---

## OpenEnv compliance

- Full `step()` / `reset()` / `state()` implementation
- Typed Pydantic models for all actions and observations
- Deterministic graders with no LLM calls
- Reproducible with fixed seed
- Dense reward across full trajectory
- Session isolation for parallel evaluation runs

---

## Project structure

```
incident-response-env/
├── server/
│   ├── app.py              # FastAPI app — /reset, /step, /state, /health
│   ├── environment.py      # Step/reset/state logic, reward computation
│   ├── incidents.py        # Deterministic incident generator (seed-based)
│   ├── simulator.py        # Dynamic log + metrics evolution
│   └── tasks.py            # Task configs (difficulty, max_steps, graders)
├── models.py               # Pydantic models (Action, Observation, State)
├── inference.py            # Agent loop (local LoRA or OpenAI-compatible API)
├── utils.py                # format_stateful_prompt, generate, parse_output
├── train.py                # GRPO training script (Unsloth + TRL)
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile              # HF Spaces deployment
├── requirements.txt        # Python dependencies
└── README.md
```

---

## License

MIT
avatar
Ask In Chat
Ask In Chat
