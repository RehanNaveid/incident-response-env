"""Focused validation for R2 candidate normalization and task-3 grading."""
import sys
sys.path.insert(0, ".")

from server.environment import IncidentResponseEnv


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"PASS {msg}")


candidates = ["auth-service", "db-primary", "cache-cluster"]
malformed = {"auth-service": 2.0, "ungraded-red-herring": 8.0}
normalized = IncidentResponseEnv._normalize_belief(malformed, candidates)

check(set(normalized.keys()) == set(candidates), "belief keys match candidates")
check(abs(sum(normalized.values()) - 1.0) < 1e-9, "belief sums to 1")
check(
    IncidentResponseEnv._compute_step_xent(malformed, "auth-service", candidates) == 1.0,
    "extra keys are ignored before R2",
)

env = IncidentResponseEnv()
env._task_id = "ambiguous_payment_degradation"
env._incident_data = {
    "root_cause": "memory_leak",
    "affected_services": ["payment-service"],
    "hypotheses": ["db_overload", "rate_limit", "memory_leak"],
}
env._step_count = 2
env._belief_trajectory = [
    {"memory_leak": 0.7, "db_overload": 0.2, "rate_limit": 0.1},
    {"memory_leak": 0.9, "db_overload": 0.05, "rate_limit": 0.05},
]

check(env._compute_r2_reward() > 0.0, "task 3 R2 grades hypotheses")
