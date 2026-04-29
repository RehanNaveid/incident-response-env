"""
Smoke test for Phase 1 — Environment Hardening.

Verifies all three atomic changes ship and interlock correctly:

  A. Fan-in DAG generator
       /incident-meta exposes fan_in_dag with nodes + edges
       DAG structure is non-trivial (has both nodes and edges)

  B. Belief → output only (not input)
       IncidentObservation has no belief_state field (was already clean)
       Belief parsed from reasoning lands in /state belief_trajectory

  C. R2 reward signal
       /state info contains r2_score in [0.0, 1.0]
       belief_brier_per_step is a list of floats in [0.0, 1.0]
       3 steps with hardcoded belief → 3 Brier entries

Run against a live server:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
    python _smoke_test.py
"""
import json
import sys
import requests

BASE = "http://localhost:7860"
FAIL = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAIL.append(msg)
        print(f"  FAIL: {msg}", flush=True)
    else:
        print(f"  PASS: {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1 — Reset with cascading_failure seed=42
# ---------------------------------------------------------------------------
print("=== RESET ===")
resp = requests.post(f"{BASE}/reset", json={"task_id": "cascading_failure", "seed": 42})
resp.raise_for_status()
reset_data = resp.json()
obs = reset_data.get("observation", {})

print("affected_services:", obs.get("affected_services"))
check("belief_state" not in obs,   "belief_state absent from reset observation")
check("fan_in_dag"   not in obs,   "fan_in_dag absent from reset observation (agent does not see it)")
check(bool(obs.get("affected_services")), "affected_services is non-empty")

# ---------------------------------------------------------------------------
# Step 2 — Three steps with explicit Belief: {...} in reasoning
# ---------------------------------------------------------------------------
affected = obs.get("affected_services", ["auth-service", "db-primary", "cache-cluster"])

steps = [
    {
        "action": f"investigate {affected[0]}",
        "reasoning": (
            f'Thought: {affected[0]} shows elevated error_rate. db-primary shows connection '
            f'pool pressure. cache-cluster shows elevated eviction rate.\n'
            f'Belief: {{"{affected[0]}": 0.6, "{affected[1] if len(affected) > 1 else "db-primary"}": 0.3, '
            f'"{affected[2] if len(affected) > 2 else "cache-cluster"}": 0.1}}'
        ),
    },
    {
        "action": f"investigate {affected[1] if len(affected) > 1 else 'db-primary'}",
        "reasoning": (
            f'Thought: investigated {affected[0]}, now checking second service. Auth errors dominant.\n'
            f'Belief: {{"{affected[0]}": 0.7, "{affected[1] if len(affected) > 1 else "db-primary"}": 0.2, '
            f'"{affected[2] if len(affected) > 2 else "cache-cluster"}": 0.1}}'
        ),
    },
    {
        "action": "assign to backend",
        "reasoning": (
            f'Thought: {affected[0]} is most likely root cause based on CRIT log. Confidence rising.\n'
            f'Belief: {{"{affected[0]}": 0.85, "{affected[1] if len(affected) > 1 else "db-primary"}": 0.1, '
            f'"{affected[2] if len(affected) > 2 else "cache-cluster"}": 0.05}}'
        ),
    },
]

print("\n=== STEPS ===")
for i, step_payload in enumerate(steps, 1):
    r = requests.post(f"{BASE}/step", json={"action": step_payload})
    r.raise_for_status()
    sdata = r.json()
    step_obs = sdata.get("observation", {})
    print(f"\n--- Step {i} ---")
    print(f"  feedback: {step_obs.get('feedback', '')}")
    check(
        "belief_state" not in step_obs,
        f"belief_state absent from step {i} observation",
    )

# ---------------------------------------------------------------------------
# Step 3 — /state: belief_trajectory, R2, Brier per step
# ---------------------------------------------------------------------------
print("\n=== /state ===")
state_resp = requests.get(f"{BASE}/state")
state_resp.raise_for_status()
state = state_resp.json()
info  = state.get("info", {})

print("info keys:", sorted(info.keys()))

# Belief trajectory
bt = info.get("belief_trajectory", [])
print(f"belief_trajectory length: {len(bt)}")
check(len(bt) == 3, f"belief_trajectory has 3 entries (got {len(bt)})")
for j, entry in enumerate(bt):
    check(isinstance(entry, dict),  f"trajectory[{j}] is a dict")
    check(all(isinstance(v, float) for v in entry.values()),
          f"trajectory[{j}] values are floats")

# R2 score
r2 = info.get("r2_score")
print(f"r2_score: {r2}")
check(r2 is not None,         "r2_score present in state info")
check(isinstance(r2, float),  "r2_score is a float")
check(0.0 <= r2 <= 1.0,       f"r2_score in [0, 1] (got {r2})")

# Brier per step
bps = info.get("belief_brier_per_step", [])
print(f"belief_brier_per_step: {bps}")
check(len(bps) == 3, f"belief_brier_per_step has 3 entries (got {len(bps)})")
for j, v in enumerate(bps):
    check(isinstance(v, float) and 0.0 <= v <= 1.0,
          f"brier_per_step[{j}]={v} in [0,1]")

# fan_in_dag in state
dag_state = info.get("fan_in_dag", {})
print(f"fan_in_dag in state: {bool(dag_state)}")
check(bool(dag_state),             "fan_in_dag present in /state info")
check("nodes" in dag_state,        "fan_in_dag.nodes in /state")
check("edges" in dag_state,        "fan_in_dag.edges in /state")
check("root"  in dag_state,        "fan_in_dag.root in /state")
check(len(dag_state.get("nodes", [])) >= 2,
      f"fan_in_dag has ≥2 nodes (got {len(dag_state.get('nodes', []))})")

# ---------------------------------------------------------------------------
# Step 4 — /incident-meta: fan_in_dag (agent-visible only)
# ---------------------------------------------------------------------------
print("\n=== /incident-meta ===")
meta_resp = requests.get(f"{BASE}/incident-meta")
meta_resp.raise_for_status()
meta = meta_resp.json()

dag_meta = meta.get("fan_in_dag", {})
print(f"fan_in_dag.nodes: {dag_meta.get('nodes')}")
print(f"fan_in_dag.edges count: {len(dag_meta.get('edges', []))}")

check(dag_meta,                          "fan_in_dag present in /incident-meta")
check("nodes" in dag_meta,               "fan_in_dag.nodes in /incident-meta")
check("edges" in dag_meta,               "fan_in_dag.edges in /incident-meta")
check("root"  not in dag_meta,           "fan_in_dag.root hidden from /incident-meta (agent cannot see ground truth)")
check(len(dag_meta.get("nodes", [])) >= 2,
      f"fan_in_dag has ≥2 nodes in /incident-meta")

# ---------------------------------------------------------------------------
# R2 boundary sanity: all-wrong belief → r2 should be low
# ---------------------------------------------------------------------------
print("\n=== R2 Boundary Check ===")
candidates = info.get("fan_in_candidates", [])
root_svc   = info.get("fan_in_candidates", ["__unknown__"])[0]  # just for type check

from server.environment import IncidentResponseEnv

# Perfect belief (prob 1.0 on correct root)
fake_incident = {
    "root_cause_service": root_svc,
    "fan_in_candidates":  candidates if candidates else [root_svc],
}
env_check = IncidentResponseEnv()
# Manually call static method
if candidates:
    perfect = {c: (1.0 if c == candidates[0] else 0.0) for c in candidates}
    worst   = {c: (0.0 if c == candidates[0] else 1.0 / max(len(candidates) - 1, 1))
               for c in candidates}
    brier_perfect = IncidentResponseEnv._compute_step_brier(perfect, candidates[0], candidates)
    brier_worst   = IncidentResponseEnv._compute_step_brier(worst,   candidates[0], candidates)
    print(f"  Brier (perfect belief): {brier_perfect:.4f}")
    print(f"  Brier (worst belief):   {brier_worst:.4f}")
    check(brier_perfect > brier_worst, "Perfect belief scores higher than worst belief")
    check(brier_perfect >= 0.5,        f"Perfect Brier ≥ 0.5 (got {brier_perfect:.4f})")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if FAIL:
    print(f"FAILED {len(FAIL)} check(s):")
    for f in FAIL:
        print(f"  • {f}")
    sys.exit(1)
else:
    print(f"All smoke tests PASS ({len(steps) * 3 + 20}+ assertions)")

print(f"\nFull /state JSON:\n{json.dumps(state, indent=2)[:4000]}")
