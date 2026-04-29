"""Offline validation for Phase 1 + Phase 2 reward shaping.

Covers:
  Phase 1 — fan-in DAG generator, Brier/stability internals
  Phase 2 — cross-entropy R2 formula, zero-score missing steps, R1+R2 scoring
"""
import math
import sys
sys.path.insert(0, ".")

from server.incidents import generate_incident
from server.environment import IncidentResponseEnv

PASS_COUNT = 0
FAIL_LIST  = []

def check(cond: bool, msg: str) -> None:
    global PASS_COUNT
    if cond:
        PASS_COUNT += 1
        print(f"  PASS  {msg}")
    else:
        FAIL_LIST.append(msg)
        print(f"  FAIL  {msg}")


# -----------------------------------------------------------------------
# 1. Fan-in DAG: seed rotation + structural invariants
# -----------------------------------------------------------------------
print("=== Phase 1 — Fan-in DAG (20 seeds) ===")
roots = set()
for s in range(20):
    inc  = generate_incident("cascading_failure", s)
    root = inc["root_cause_service"]
    roots.add(root)
    dag  = inc["fan_in_dag"]

    check(dag["root"] == root,            f"seed {s}: dag.root matches incident root")
    check(len(dag["nodes"]) >= 2,         f"seed {s}: dag has ≥2 nodes")
    check(bool(dag["spurious_edges"]),     f"seed {s}: spurious edges injected")
    check(bool(dag["missing_edges"]),      f"seed {s}: edges dropped (partial observability)")
    check(root in dag["nodes"],            f"seed {s}: root in nodes")

    obs_set      = set(tuple(e) for e in dag["edges"])
    spurious_set = set(tuple(e) for e in dag["spurious_edges"])
    missing_set  = set(tuple(e) for e in dag["missing_edges"])
    true_set     = set(tuple(e) for e in dag["true_edges"])

    for e in spurious_set:
        check(e in obs_set,       f"seed {s}: spurious {e} in observable")
        check(e not in true_set,  f"seed {s}: spurious {e} not in true_edges")
    for e in missing_set:
        check(e not in obs_set,   f"seed {s}: dropped {e} absent from observable")

print(f"\n  Distinct roots: {sorted(roots)}")
check(len(roots) >= 4, f"≥4 distinct roots across 20 seeds (got {len(roots)})")

# -----------------------------------------------------------------------
# 2. Phase 2 — cross-entropy per-step reward
# -----------------------------------------------------------------------
print("\n=== Phase 2 — Cross-entropy step reward ===")
candidates = ["auth-service", "db-primary", "cache-cluster"]
root       = "auth-service"
N          = len(candidates)

# Perfect belief: p̂(root)=1.0 → reward = 1.0
perfect = {"auth-service": 1.0, "db-primary": 0.0, "cache-cluster": 0.0}
xp = IncidentResponseEnv._compute_step_xent(perfect, root, candidates)
print(f"  Perfect belief  xent={xp:.6f}")
check(abs(xp - 1.0) < 1e-9, "Perfect belief → xent=1.0")

# Uniform belief: p̂(root)=1/N → reward = 0.0
uniform = {c: 1.0/N for c in candidates}
xu = IncidentResponseEnv._compute_step_xent(uniform, root, candidates)
print(f"  Uniform belief  xent={xu:.6f}  (expected 0.0)")
check(abs(xu) < 1e-9, "Uniform belief → xent=0.0")

# Mid belief: p̂(root)=0.7 → reward = 1 + log(0.7)/log(3)
xm_expected = max(0.0, 1.0 + math.log(0.7) / math.log(N))
mid = {"auth-service": 0.7, "db-primary": 0.2, "cache-cluster": 0.1}
xm = IncidentResponseEnv._compute_step_xent(mid, root, candidates)
print(f"  Mid belief      xent={xm:.6f}  (expected {xm_expected:.6f})")
check(abs(xm - xm_expected) < 1e-6, f"Mid belief → xent≈{xm_expected:.4f}")

# Worst belief: p̂(root)→0 → reward = 0 (clamped)
worst = {"auth-service": 0.01, "db-primary": 0.5, "cache-cluster": 0.49}
xw = IncidentResponseEnv._compute_step_xent(worst, root, candidates)
print(f"  Worst belief    xent={xw:.6f}  (expected ≈0)")
check(xw >= 0.0,  "Worst belief → xent ≥ 0 (clamped)")
check(xw < 0.1,   f"Worst belief → xent < 0.1 (got {xw:.4f})")

# Ordering: perfect > mid > uniform > worst
check(xp > xm > xu >= xw, f"Ordering: perfect({xp:.3f}) > mid({xm:.3f}) > uniform({xu:.3f}) ≥ worst({xw:.3f})")

# -----------------------------------------------------------------------
# 3. Phase 2 — R2 zero-score missing steps
# -----------------------------------------------------------------------
print("\n=== Phase 2 — R2 zero-score missing steps ===")

env = IncidentResponseEnv()
env._task_id = "cascading_failure"
env._incident_data = {
    "root_cause_service": "auth-service",
    "fan_in_candidates":  ["auth-service", "db-primary", "cache-cluster"],
}

# 3 beliefs, 3 steps total → R2 = avg(xent) / 1.0
env._step_count        = 3
env._belief_trajectory = [
    {"auth-service": 1.0, "db-primary": 0.0, "cache-cluster": 0.0},  # xent=1.0
    {"auth-service": 1.0, "db-primary": 0.0, "cache-cluster": 0.0},  # xent=1.0
    {"auth-service": 1.0, "db-primary": 0.0, "cache-cluster": 0.0},  # xent=1.0
]
r2_all_perfect = env._compute_r2_reward()
print(f"  3/3 perfect beliefs, 3 steps → R2={r2_all_perfect:.4f}")
check(abs(r2_all_perfect - 1.0) < 1e-9, "3/3 perfect beliefs → R2=1.0")

# 2 beliefs reported, 6 steps total → R2 = 2*1.0 / 6
env._step_count        = 6
env._belief_trajectory = [
    {"auth-service": 1.0, "db-primary": 0.0, "cache-cluster": 0.0},
    {"auth-service": 1.0, "db-primary": 0.0, "cache-cluster": 0.0},
]
r2_partial = env._compute_r2_reward()
expected_partial = 2.0 / 6
print(f"  2/6 perfect beliefs → R2={r2_partial:.4f}  (expected {expected_partial:.4f})")
check(abs(r2_partial - expected_partial) < 1e-9, f"2/6 beliefs → R2={expected_partial:.4f} (zero-score missing)")

# 0 beliefs reported → R2 = 0
env._belief_trajectory = []
r2_none = env._compute_r2_reward()
print(f"  0 beliefs → R2={r2_none:.4f}")
check(r2_none == 0.0, "No beliefs → R2=0.0")

# -----------------------------------------------------------------------
# 4. Phase 2 — Stability reward
# -----------------------------------------------------------------------
print("\n=== Phase 2 — Stability reward (KL penalty) ===")
env2 = IncidentResponseEnv()
env2._belief_trajectory = [
    {"auth-service": 0.6, "db-primary": 0.3, "cache-cluster": 0.1},
    {"auth-service": 0.62,"db-primary": 0.28,"cache-cluster": 0.1},  # tiny shift
]
stable = env2._compute_stability_reward()

env2._belief_trajectory = [
    {"auth-service": 0.9, "db-primary": 0.05,"cache-cluster": 0.05},
    {"auth-service": 0.05,"db-primary": 0.9, "cache-cluster": 0.05},  # wild flip
]
unstable = env2._compute_stability_reward()

print(f"  stable penalty={stable:.6f}  unstable penalty={unstable:.6f}")
check(stable > unstable,   "Stable penalised less than unstable")
check(stable  <= 0.0,      "Stability penalty ≤ 0")
check(unstable >= -0.10,   "Unstable penalty ≥ -0.10 (bounded)")

# -----------------------------------------------------------------------
# 5. Phase 2 — Final scoring formula sanity
# -----------------------------------------------------------------------
print("\n=== Phase 2 — Final score formula (R1*0.60 + R2*0.40) ===")

from inference import R1_WEIGHT, R2_WEIGHT
check(abs(R1_WEIGHT - 0.60) < 1e-9, f"R1_WEIGHT=0.60 (got {R1_WEIGHT})")
check(abs(R2_WEIGHT - 0.40) < 1e-9, f"R2_WEIGHT=0.40 (got {R2_WEIGHT})")

# Test blended score
r1, r2 = 0.75, 0.50
blended = R1_WEIGHT * r1 + R2_WEIGHT * r2
expected_blend = 0.60 * 0.75 + 0.40 * 0.50
check(abs(blended - expected_blend) < 1e-9,
      f"Blended score: 0.60*{r1}+0.40*{r2}={blended:.4f}")

# Perfect R1, no belief → score = 0.60 * 1.0 + 0.40 * 0.0 = 0.60
score_no_belief = R1_WEIGHT * 1.0 + R2_WEIGHT * 0.0
print(f"  Perfect R1, no belief → score={score_no_belief:.2f}")
check(abs(score_no_belief - 0.60) < 1e-9,
      "Perfect task + no belief = 0.60 (agent penalised for missing belief)")

# Perfect R1 + perfect R2 → score = 1.0
score_perfect = R1_WEIGHT * 1.0 + R2_WEIGHT * 1.0
check(abs(score_perfect - 1.0) < 1e-9, "Perfect R1 + perfect R2 = 1.0")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
total = PASS_COUNT + len(FAIL_LIST)
if FAIL_LIST:
    print(f"FAILED {len(FAIL_LIST)}/{total} checks:")
    for f in FAIL_LIST:
        print(f"  • {f}")
    sys.exit(1)
else:
    print(f"All {PASS_COUNT} checks PASS")
