"""Offline validation script for Phase 1 changes — no server required."""
import sys
sys.path.insert(0, ".")

from server.incidents import generate_incident, build_fan_in_dag
from server.environment import IncidentResponseEnv

# -----------------------------------------------------------------------
# 1. Fan-in DAG: 20-seed rotation
# -----------------------------------------------------------------------
print("=== Fan-in DAG: 20-seed rotation ===")
roots = set()
for s in range(20):
    inc  = generate_incident("cascading_failure", s)
    root = inc["root_cause_service"]
    roots.add(root)
    dag  = inc["fan_in_dag"]

    assert dag["root"] == root,        f"seed {s}: dag.root mismatch"
    assert len(dag["nodes"]) >= 2,     f"seed {s}: too few nodes"
    assert isinstance(dag["edges"], list), f"seed {s}: edges not list"
    assert dag["spurious_edges"],      f"seed {s}: no spurious edges injected"
    assert dag["missing_edges"],       f"seed {s}: no edges dropped"
    assert root in dag["nodes"],       f"seed {s}: root not in nodes"

    obs_set      = set(tuple(e) for e in dag["edges"])
    spurious_set = set(tuple(e) for e in dag["spurious_edges"])
    missing_set  = set(tuple(e) for e in dag["missing_edges"])
    true_set     = set(tuple(e) for e in dag["true_edges"])

    for e in spurious_set:
        assert e in obs_set,      f"seed {s}: spurious edge {e} not in observable"
        assert e not in true_set, f"seed {s}: spurious edge {e} in true_edges"

    for e in missing_set:
        assert e not in obs_set,  f"seed {s}: dropped edge {e} still in observable"

    print(
        f"  seed={s:2d}  root={root!s:<22}  "
        f"nodes={len(dag['nodes'])}  edges={len(dag['edges'])}  "
        f"spurious={len(dag['spurious_edges'])}  missing={len(dag['missing_edges'])}"
    )

assert len(roots) >= 4, f"Only {len(roots)} distinct roots: {roots}"
print(f"\nDistinct roots ({len(roots)}): {sorted(roots)}")

# -----------------------------------------------------------------------
# 2. Brier boundary check
# -----------------------------------------------------------------------
print("\n=== Brier boundary check ===")
candidates = ["auth-service", "db-primary", "cache-cluster"]
root = "auth-service"

perfect = {"auth-service": 1.0, "db-primary": 0.0, "cache-cluster": 0.0}
worst   = {"auth-service": 0.0, "db-primary": 0.5, "cache-cluster": 0.5}
uniform = {"auth-service": 0.33, "db-primary": 0.33, "cache-cluster": 0.34}

bp = IncidentResponseEnv._compute_step_brier(perfect, root, candidates)
bu = IncidentResponseEnv._compute_step_brier(uniform, root, candidates)
bw = IncidentResponseEnv._compute_step_brier(worst,   root, candidates)

print(f"  perfect Brier={bp:.4f}")
print(f"  uniform Brier={bu:.4f}")
print(f"  worst   Brier={bw:.4f}")

assert bp > bu > bw,  f"Ordering violated: {bp} > {bu} > {bw}"
assert bp >= 0.6,     f"Perfect Brier too low: {bp}"
assert bw <= 0.5,     f"Worst Brier too high: {bw}"
print("  Ordering: perfect > uniform > worst  PASS")

# -----------------------------------------------------------------------
# 3. Stability reward: stable vs oscillating
# -----------------------------------------------------------------------
print("\n=== Stability reward ===")
import math

env_tmp = IncidentResponseEnv()
env_tmp._belief_trajectory = [
    {"auth-service": 0.6, "db-primary": 0.3, "cache-cluster": 0.1},
    {"auth-service": 0.65, "db-primary": 0.25, "cache-cluster": 0.1},   # small shift
]
stable_penalty = env_tmp._compute_stability_reward()

env_tmp._belief_trajectory = [
    {"auth-service": 0.9, "db-primary": 0.05, "cache-cluster": 0.05},
    {"auth-service": 0.05, "db-primary": 0.9, "cache-cluster": 0.05},  # wild flip
]
unstable_penalty = env_tmp._compute_stability_reward()

print(f"  stable belief penalty:   {stable_penalty:.4f}")
print(f"  unstable belief penalty: {unstable_penalty:.4f}")
assert stable_penalty > unstable_penalty, "Stable should be penalised less"
assert stable_penalty <= 0.0,            "Stability reward must be <= 0"
assert unstable_penalty <= 0.0,          "Stability reward must be <= 0"
print("  Stable penalised less than unstable  PASS")

# -----------------------------------------------------------------------
# 4. R2 trajectory (full episode)
# -----------------------------------------------------------------------
print("\n=== R2 trajectory ===")
env_r2 = IncidentResponseEnv()
env_r2._task_id = "cascading_failure"
env_r2._incident_data = {
    "root_cause_service": "auth-service",
    "fan_in_candidates":  ["auth-service", "db-primary", "cache-cluster"],
}
# Converging trajectory
env_r2._belief_trajectory = [
    {"auth-service": 0.4, "db-primary": 0.4, "cache-cluster": 0.2},
    {"auth-service": 0.6, "db-primary": 0.3, "cache-cluster": 0.1},
    {"auth-service": 0.85,"db-primary": 0.1, "cache-cluster": 0.05},
]
r2_converging = env_r2._compute_r2_reward()

# Flat-wrong trajectory
env_r2._belief_trajectory = [
    {"auth-service": 0.05, "db-primary": 0.9, "cache-cluster": 0.05},
    {"auth-service": 0.05, "db-primary": 0.9, "cache-cluster": 0.05},
    {"auth-service": 0.05, "db-primary": 0.9, "cache-cluster": 0.05},
]
r2_wrong = env_r2._compute_r2_reward()

print(f"  R2 (converging to correct root): {r2_converging:.4f}")
print(f"  R2 (wrong root throughout):      {r2_wrong:.4f}")
assert r2_converging > r2_wrong, "Converging trajectory must score higher"
assert 0.0 <= r2_converging <= 1.0
assert 0.0 <= r2_wrong      <= 1.0
print("  Converging > wrong  PASS")

print("\n=== All offline checks PASS ===")
