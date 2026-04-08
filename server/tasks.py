"""
Task definitions and grading for the Incident Response Environment.

Every grader is **pure Python** – no LLM calls.  Scoring is based on keyword
matching against the agent's action history and incident seed_meta.

Three task graders:
    - grade_task1  →  single_service_outage
    - grade_task2  →  cascading_failure
    - grade_task3  →  ambiguous_payment_degradation

Each grader signature:  (action_history: List[str], seed_meta: dict) → float [0.0–1.0]
"""

from typing import Any, Dict, List


def _parse_category(action: str) -> str:
    a = action.lower()
    if "investigate" in a:
        return "investigate"
    if "mitigate" in a or "rollback" in a or "restart" in a:
        return "mitigate"
    if "resolve" in a:
        return "resolve"
    if "assign" in a:
        return "assign"
    return "other"


def _score_reasoning(reasoning_texts: list[str]) -> float:
    if not reasoning_texts:
        return -0.1  # no reasoning penalty

    text = " ".join(reasoning_texts).lower()

    score = 0.0

    # Evidence-based reasoning
    if "connection" in text or "db" in text:
        score += 0.1
    if "latency" in text or "timeout" in text:
        score += 0.1
    if "retry" in text or "queue" in text:
        score += 0.1

    # Causal reasoning
    if "because" in text or "due to" in text:
        score += 0.1

    # Hypothesis thinking
    if "possible" in text or "might" in text or "likely" in text:
        score += 0.1

    # Penalize shallow reasoning
    if len(text.split()) < 5:
        score -= 0.1

    return score


def _score_reasoning_evolution(reasoning_texts: list[str]) -> float:
    if not reasoning_texts or len(reasoning_texts) < 2:
        return -0.1  # no evolution

    stages = {
        "hypothesis": 0,
        "evidence": 0,
        "conclusion": 0,
    }

    for text in reasoning_texts:
        t = text.lower()

        # hypothesis signals
        if any(w in t for w in ["maybe", "might", "possible", "likely"]):
            stages["hypothesis"] += 1

        # evidence signals
        if any(w in t for w in ["log", "metric", "latency", "connection", "retry"]):
            stages["evidence"] += 1

        # conclusion signals
        if any(w in t for w in ["therefore", "root cause", "hence", "so"]):
            stages["conclusion"] += 1

    score = 0.0

    # reward progression across stages
    if stages["hypothesis"] > 0:
        score += 0.1
    if stages["evidence"] > 0:
        score += 0.1
    if stages["conclusion"] > 0:
        score += 0.15

    # penalize repetition (same reasoning every step)
    unique_reasoning = len(set(r.strip().lower() for r in reasoning_texts))
    if unique_reasoning <= 2:
        score -= 0.1

    return score


# ===================================================================
# Task-level graders
# ===================================================================


def grade_task1(action_history: List[str], seed_meta: dict, reasoning_texts=None) -> float:
    """Grader for **single_service_outage**.

    +0.25  correct team assigned (from seed_meta, not hard-coded)
    +0.30  correct mitigation keyword used
    +0.15  investigated before mitigating
    +0.20  resolved
    +0.10  efficiency bonus (≤ 6 steps)
    """
    actions = " ".join(action_history).lower()
    score = 0.0

    # Correct team from incident metadata (not hard-coded "backend")
    correct_team = seed_meta.get("correct_team", "backend").lower()
    if any("assign" in a.lower() and correct_team in a.lower() for a in action_history):
        score += 0.25

    # Mitigation — check against actual valid mitigations from incident
    valid_mits = seed_meta.get("valid_mitigations", ["restart", "pool", "connection"])
    mitigation_done = any(
        "mitigate" in a.lower() and any(kw in a.lower() for kw in valid_mits)
        for a in action_history
    )
    if mitigation_done:
        score += 0.30

    # Must investigate the actual affected service
    affected = seed_meta.get("affected_services", ["auth-service"])
    investigated = any(
        "investigate" in a.lower() and any(svc.lower() in a.lower() for svc in affected)
        for a in action_history
    )
    if investigated:
        score += 0.15

    # Resolution
    if mitigation_done and any("resolve" in a.lower() for a in action_history):
        score += 0.20

    # Efficiency bonus
    if investigated and mitigation_done and len(action_history) <= 6:
        score += 0.10

    # Runbook usage bonus — reward agents that query available tooling
    runbook_queries = seed_meta.get("runbook_queries", [])
    if any(svc in runbook_queries for svc in affected):
        score += 0.05

    if reasoning_texts:
        score += _score_reasoning(reasoning_texts)

    # Penalize repeated identical mitigations
    mitigate_actions = [a for a in action_history if "mitigate" in a.lower()]
    unique_mitigations = len(set(a.lower() for a in mitigate_actions))

    repeat_penalty = max(0, len(mitigate_actions) - unique_mitigations) * 0.08
    score -= repeat_penalty

    score = max(0.0, min(score, 1.0))
    return score


def grade_task2(action_history: List[str], seed_meta: dict, reasoning_texts=None) -> float:
    score = 0.0

    required_services = seed_meta.get("affected_services", [])
    valid_mitigations = seed_meta.get("valid_mitigations", [])
    correct_team = (seed_meta.get("correct_team") or "").lower().strip()
    root_cause_service = (seed_meta.get("root_cause_service") or "").lower().strip()

    if not required_services or not valid_mitigations:
        return 0.0

    # --- Investigation score (max 0.45) ---
    per_service = 0.45 / len(required_services)
    investigated_services = []
    for svc in required_services:
        if any(
            "investigate" in a.lower() and svc.lower() in a.lower()
            for a in action_history
        ):
            score += per_service
            investigated_services.append(svc)

    all_investigated = len(investigated_services) == len(required_services)

    # --- Root cause service bonus (0.10) ---
    if root_cause_service:
        if any(
            "investigate" in a.lower() and root_cause_service in a.lower()
            for a in action_history
        ):
            score += 0.10

    # --- Mitigation (0.25 full, 0.05 partial) ---
    # Match if ANY valid mitigation keyword appears in ANY action
    mitigate_actions = [
        a for a in action_history
        if any(kw.lower() in a.lower() for kw in valid_mitigations)
    ]
    mitigation_done = len(mitigate_actions) > 0

    if all_investigated and mitigation_done:
        score += 0.25
    elif mitigation_done:
        score += 0.05

    # --- Team assignment ---
    any_assign = any("assign" in a.lower() for a in action_history)
    correct_assign = any(
        "assign" in a.lower() and correct_team and correct_team in a.lower()
        for a in action_history
    )

    if correct_assign:
        score += 0.10
    elif any_assign:
        score -= 0.05   # wrong team
    else:
        score -= 0.10   # skipped entirely

    # --- Efficiency (0.10) ---
    if all_investigated and mitigation_done and len(action_history) <= 12:
        score += 0.10

    # --- Resolve (0.05) ---
    if mitigation_done and any("resolve" in a.lower() for a in action_history):
        score += 0.05

    # --- Repeat mitigation penalty ---
    unique_mit = len(set(a.lower() for a in mitigate_actions))
    repeats = len(mitigate_actions) - unique_mit
    if repeats > 1:
        score -= (repeats - 1) * 0.10  # one repeat is tolerated, rest penalized

    # --- Premature resolve penalty ---
    actions_lower = [a.lower() for a in action_history]
    first_resolve_idx = next(
        (i for i, a in enumerate(actions_lower) if "resolve" in a), None
    )
    first_mit_idx = next(
        (i for i, a in enumerate(actions_lower)
         if any(kw.lower() in a for kw in valid_mitigations)),
        None
    )
    if (first_resolve_idx is not None
            and first_mit_idx is not None
            and first_resolve_idx <= first_mit_idx):
        score *= 0.50

    # --- Reasoning bonus ---
    if reasoning_texts:
        score += _score_reasoning(reasoning_texts)

    # --- Runbook bonus (0.05) ---
    runbook_queries = seed_meta.get("runbook_queries", [])
    if any(svc in runbook_queries for svc in required_services):
        score += 0.05

    return max(0.0, min(score, 1.0))

def grade_task3(action_history: List[str], seed_meta: dict, reasoning_texts=None) -> float:
    """Grader for **ambiguous_payment_degradation**.

    +0.10  each hypothesis investigated in a separate step (max 0.30)
    +0.25  correct mitigation keyword used in mitigate/assign/fix action
    +0.20  correct team assigned
    +0.15  resolved
    +0.10  efficiency (≤ 15 steps)
    """
    score = 0.0

    # Hypothesis investigation — must contain "investigate" + specific keyword
    hypothesis_keywords = {
        "db_overload": ["db", "database", "connection", "query", "deadlock"],
        "rate_limit":  ["rate", "stripe", "429", "throttle", "upstream"],
        "memory_leak": ["memory", "heap", "oom", "leak", "gc"],
    }
    investigated_hypotheses = set()
    for action in action_history:
        a = action.lower()
        if "investigate" not in a:
            continue
        for hypothesis, keywords in hypothesis_keywords.items():
            if any(kw in a for kw in keywords):
                investigated_hypotheses.add(hypothesis)

    score += min(0.30, len(investigated_hypotheses) * 0.10)

    # Root cause — check correct mitigation keywords used in action context
    valid_mitigations = seed_meta.get("valid_mitigations", [])
    root_cause_found = any(
        any(kw in a.lower() for kw in valid_mitigations)
        and any(v in a.lower() for v in [
            "mitigate", "assign", "fix", "restart",
            "rollback", "throttle", "vacuum", "backoff",
        ])
        for a in action_history
    )
    if root_cause_found:
        score += 0.25

    # Correct team assignment
    team = seed_meta.get(
        "correct_team",
        seed_meta.get("team", "payments-oncall"),
    ).lower()
    team_assigned = any(
        team in a.lower() and "assign" in a.lower()
        for a in action_history
    )
    if team_assigned:
        score += 0.20

    # Resolution
    if any("resolve" in a.lower() for a in action_history):
        score += 0.15

    # Efficiency
    if len(action_history) <= 15:
        score += 0.10

    # --- Reasoning field integration ---
    # Award partial credit if the agent's reasoning mentions hypothesis
    # keywords even without explicit investigate actions.  This makes
    # the environment evaluate chain-of-thought, a unique differentiator.
    reasoning_text = seed_meta.get("agent_reasoning", "").lower()
    if reasoning_text:
        for hyp, keywords in hypothesis_keywords.items():
            if hyp not in investigated_hypotheses:
                if any(kw in reasoning_text for kw in keywords):
                    investigated_hypotheses.add(hyp)
                    score += 0.05  # smaller credit for reasoning-only

    # Runbook usage bonus
    runbook_queries = seed_meta.get("runbook_queries", [])
    affected = seed_meta.get("affected_services", ["payment-service"])
    if any(svc in runbook_queries for svc in affected):
        score += 0.05

    # Count investigated hypotheses
    # hypothesis_count = 0
    # investigate_actions = [a for a in action_history if "investigate" in a.lower()]
    # if any("db" in a.lower() or "database" in a.lower() or "connection" in a.lower() 
    #     for a in investigate_actions):
    #     hypothesis_count += 1
    # if any("rate" in a.lower() or "stripe" in a.lower() or "throttle" in a.lower()
    #     for a in investigate_actions):
    #     hypothesis_count += 1
    # if any("memory" in a.lower() or "heap" in a.lower() or "oom" in a.lower()
    #     for a in investigate_actions):
    #     hypothesis_count += 1
    hypothesis_count = len(investigated_hypotheses)
    if hypothesis_count == 0:
        score -= 0.20
    elif hypothesis_count == 1:
        score -= 0.10

    # # Penalize shallow reasoning
    # if hypothesis_count == 0:
    #     score -= 0.15  # flat deduction, not a multiplier
    # elif hypothesis_count == 1:
    #     score -= 0.05 

    if reasoning_texts:
        score += _score_reasoning(reasoning_texts)

    if reasoning_texts:
        text = " ".join(reasoning_texts).lower()

        # Reward correct inference chain
        if "connection" in text and "db" in text:
            score += 0.15

        # Penalize blind runbook following
        if "restart" in text and "db" not in text:
            score -= 0.1

    if reasoning_texts:
        evolution_score = _score_reasoning_evolution(reasoning_texts)
        score += evolution_score

    score = max(0.0, min(score, 1.0))
    return score


# ===================================================================
# Grader runner helper
# ===================================================================

def run_grader(task_id: str, action_history: List[str],
               seed_meta: dict) -> float:
    """Run the grader for a task and return score 0.0–1.0."""
    config = TASK_CONFIGS.get(task_id)
    if not config:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    return config["grader"](action_history, seed_meta)


# ===================================================================
# TASK_CONFIGS
# ===================================================================

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "single_service_outage": {
        "grader": grade_task1,
        "max_steps": 12,               # was 10
        "max_reward": 1.00,
        "difficulty": "easy",
        "description": "Single service outage. Investigate the affected service, assign the owner, mitigate from logs, resolve.",
    },
    "cascading_failure": {
        "grader": grade_task2,
        "max_steps": 18,
        "max_reward": 1.00,
        "difficulty": "medium",
        "description": "Three-service cascade. Investigate every affected service, identify the root cause, mitigate from logs, resolve.",
    },
    "ambiguous_payment_degradation": {
        "grader": grade_task3,
        "max_steps": 25,
        "max_reward": 1.00,
        "difficulty": "hard",
        "description": "Payment degraded. Investigate hypotheses, find root cause, fix.",
    },
}


# ===================================================================
# Helpers
# ===================================================================

def get_tasks() -> List[Dict[str, Any]]:
    """Return task configs as a list of dicts (each with task_id key)."""
    return [
        {
            "task_id": k,
            "max_steps": v["max_steps"],
            "max_reward": v["max_reward"],
            "difficulty": v["difficulty"],
            "description": v["description"],
        }
        for k, v in TASK_CONFIGS.items()
    ]
