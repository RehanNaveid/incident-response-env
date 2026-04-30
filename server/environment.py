"""
Incident Response Environment – OpenEnv Environment implementation.

Subclasses ``openenv.core.env_server.Environment`` and implements the
``reset()``, ``step()``, and ``state`` interface.

**No LLM calls** are made inside this module – all reward computation is
pure-Python arithmetic on outcome dicts derived from keyword matching.

Data flow (Fix 8):
    reset(task_id, seed) → generate_incident(task_id, seed) → self._incident_data
    step(action) → grade against self._incident_data["valid_mitigations"]

    There is ONE data system.  The generated incident IS the grading source.
"""

from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server import Environment

from models import IncidentAction, IncidentObservation, IncidentState
from server.incidents import generate_incident
from server.simulator import IncidentSimulator
from server.tasks import TASK_CONFIGS


# ---------------------------------------------------------------------------
# Belief calibration reward weights
# ---------------------------------------------------------------------------

R2_WEIGHT        = 0.40   # cross-entropy calibration per step (raised from 0.15)
STABILITY_WEIGHT = 0.05   # KL-divergence stability penalty (applied directly)
STEP_COST        = 0.05   # small pressure to resolve efficiently
TERMINAL_SUCCESS_BONUS = 1.00
IR_DEBUG         = os.environ.get("IR_DEBUG", "0") == "1"


def _debug(message: str) -> None:
    if IR_DEBUG:
        print(f"[DBG] {message}", flush=True)


# Canonical action categories returned by _parse_action
_ACTION_KEYWORDS: Dict[str, List[str]] = {
    "investigate": ["investigate", "analyze", "examine", "inspect", "check", "look",
                    "review", "scan", "monitor", "capture", "log", "audit"],
    "assign":      ["assign", "delegate", "task", "dispatch", "alert", "route to", "page", "contact team", "notify team"],
    "escalate":    ["escalate", "notify", "report", "contact", "call", "urgent", "critical escalation", "page oncall"],
    "mitigate":    ["mitigate", "block", "isolate", "quarantine", "contain",
                    "disable", "shut down", "stop", "kill", "remove", "drop",
                    "firewall", "blacklist", "deny", "restart", "pool",
                    "rollback", "revert", "throttle", "backoff", "vacuum",
                    "scale", "connections", "query", "rebuild", "index",
                    "purge", "cache", "batch", "pods", "canary", "config",
                    "memory", "heap"],
    "resolve":     ["resolve", "close", "remediate", "fix", "patch", "restore",
                    "recover", "rotate", "reset credentials", "rotate credentials"],
}


class IncidentResponseEnv(Environment):
    """Simulated incident-response environment.

    The agent receives an incident scenario and must choose actions to
    investigate, contain, and resolve it.  Grading uses pure-Python
    keyword matching and arithmetic – **no LLM calls**.

    Lifecycle::

        obs = env.reset(task_id="single_service_outage", seed=42)
        while not obs.done:
            obs = env.step(IncidentAction(action="investigate backend"))
    """

    def __init__(self) -> None:
        super().__init__()
        self._seed: int = 0
        self._task_id: str = "single_service_outage"
        self._incident_data: Dict[str, Any] = {}  # populated by reset()
        self._simulator: Optional[IncidentSimulator] = None
        self._initialized: bool = False

        self._consecutive_invalid: int = 0
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._resolved: bool = False
        self._mitigate_done: bool = False
        self.root_cause_identified: bool = False
        self.mitigation_attempts: int = 0
        self.improvement_streak: int = 0
        self.hypotheses: Dict[str, bool] = {"db": False, "rate": False, "memory": False}
        self._investigated: bool = False
        self._assigned: bool = False
        self._last_action: Optional[str] = None
        self._action_history: List[str] = []
        self._reasoning_texts: List[str] = []   # agent reasoning per step
        self._runbook_queries: List[str] = []    # services queried via /runbook
        self._episode_id: str = str(uuid4())
        self._latest_logs: List[str] = []
        self._latest_metrics: List[Dict[str, Any]] = []
        # Belief trajectory — populated from agent reasoning each step
        self._belief_trajectory: List[Dict[str, float]] = []
        self._last_reported_belief: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "single_service_outage",
        **kwargs: Any,
    ) -> IncidentObservation:
        """Reset the environment and present an incident.

        Args:
            seed: Deterministic seed.  If *None*, a random int is generated.
            episode_id: Optional custom episode identifier.
            task_id: Task to run.  Must be a key in TASK_CONFIGS.

        Returns:
            Initial :class:`IncidentObservation` with all fields populated.
        """
        # 1. Store task_id
        self._task_id = task_id

        # 2. Store / generate seed and seed the RNG immediately
        self._seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        random.seed(self._seed)
        self._initialized = False

        # 3. Generate incident and attach the live simulator
        self._incident_data = generate_incident(self._task_id, self._seed)
        self._simulator = IncidentSimulator(self._incident_data, self._seed)

        # 4. Reset accumulators
        self._consecutive_invalid = 0
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._resolved = False
        self._mitigate_done = False
        self.root_cause_identified = False
        self.mitigation_attempts = 0
        self.improvement_streak = 0
        self.hypotheses = {"db": False, "rate": False, "memory": False}
        self._investigated = False
        self._assigned = False
        self._last_action = None
        self._action_history = []
        self._reasoning_texts = []
        self._runbook_queries = []
        self._episode_id = episode_id or str(uuid4())
        self._latest_logs = list(self._incident_data.get("logs", [])[-8:])
        self._latest_metrics = (
            self._simulator.current_metrics() if self._simulator else []
        )
        self._belief_trajectory = []
        self._last_reported_belief = {}
        self._initialized = True

        # 5. Return fully-populated observation
        return self._build_observation(
            feedback="", score=0.0, outcome={},
            reward=0.0, done=False, metadata={},
            logs_override=self._latest_logs,
            metrics_override=self._latest_metrics,
        )

    def _ensure_initialized(self) -> None:
        """Require reset() before executing stateful environment actions."""
        if not self._initialized or not self._incident_data or self._simulator is None:
            raise RuntimeError(
                "Call reset() before step(). IncidentResponseEnv is not initialized."
            )

    def _can_resolve(self) -> bool:
        """True when a valid mitigation has been applied and services recovered."""
        if not self._mitigate_done:
            return False
        if self._simulator is None:
            return True
        return self._simulator.is_fully_recovered()

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """Process one agent action.

        Pipeline:
            1. ``_parse_action(text)`` → canonical category
            2. ``_compute_outcome(parsed)`` → outcome dict
            3. ``_compute_step_reward(parsed, outcome)`` → float
            4. Inline keyword grading against generated incident data
            5. Increment step count, check termination
            6. Return :class:`StepResult`

        No LLM calls are made.
        """
        self._ensure_initialized()
        raw_text = action.action.lower()
        self._last_action = action.action
        self._action_history.append(action.action)

        # Store reasoning for grader access
        if action.reasoning:
            self._reasoning_texts.append(action.reasoning)

        # Parse and store agent-reported belief for R2 trajectory grading
        parsed_belief = self._parse_belief_from_reasoning(action.reasoning or "")
        step_r2_reward        = 0.0
        step_stability_reward = 0.0
        if parsed_belief:
            _cands = self._belief_candidates()
            parsed_belief = self._normalize_belief(parsed_belief, _cands)
            # Compute stability BEFORE appending (compares new vs previous)
            if self._belief_trajectory:
                self._belief_trajectory.append(parsed_belief)
                step_stability_reward = self._compute_stability_reward()
            else:
                self._belief_trajectory.append(parsed_belief)
            self._last_reported_belief = parsed_belief
            # Per-step cross-entropy contribution toward R2
            # Resolution chain: fan_in_candidates → hypotheses → affected_services
            # Task 3 ambiguity is between HYPOTHESES (db_overload/rate_limit/memory_leak),
            # not services. Without 'hypotheses' in the chain, candidates = ['payment-service']
            # → len=1 → R2 silently returns 0 for every task 3 episode.
            _rc    = (
                self._incident_data.get("root_cause_service")
                or self._incident_data.get("root_cause", "")
            )
            # Only meaningful when discriminating between >1 candidate
            if _rc and _cands and len(_cands) > 1:
                step_r2_reward = self._compute_step_xent(
                    parsed_belief, _rc, _cands
                )
        # If None: no belief this step — calibration rewards stay 0.0

            _debug(
                "r2 "
                f"task={self._task_id} step={self._step_count + 1} "
                f"root={_rc!r} candidates={_cands} "
                f"belief={parsed_belief} xent={step_r2_reward:.4f} "
                f"stability={step_stability_reward:.4f}"
            )
        else:
            _debug(
                "r2 "
                f"task={self._task_id} step={self._step_count + 1} "
                "belief_parse=None xent=0.0000"
            )

        # 1. Parse
        parsed_action = self._parse_action(raw_text)

        # Root cause detection
        if parsed_action == "investigate" and ("db" in raw_text or "connection" in raw_text):
            self.root_cause_identified = True

        # Track mitigations
        if parsed_action == "mitigate":
            self.mitigation_attempts += 1

        # Hypothesis tracking (Task 3)
        if self._task_id == "ambiguous_payment_degradation" and parsed_action == "investigate":
            if "db" in raw_text:
                self.hypotheses["db"] = True
            if "rate" in raw_text:
                self.hypotheses["rate"] = True
            if "memory" in raw_text:
                self.hypotheses["memory"] = True

        # 2. Outcome
        outcome = self._compute_outcome(parsed_action)

        # 5. Inline keyword grading against generated incident data
        valid_mitigations: List[str] = self._incident_data.get("valid_mitigations", [])
        action_lower = action.action.lower()
        matched = [kw for kw in valid_mitigations if kw.lower() in action_lower]
        keyword_score = len(matched) / len(valid_mitigations) if valid_mitigations else 0.0
        correct_team = self._incident_data.get("correct_team", "").lower()
        available_teams = {
            team.lower() for team in self._incident_data.get("team_map", {}).values()
        }
        assigned_team_valid = (
            parsed_action == "assign"
            and bool(correct_team)
            and correct_team in action_lower
        )

        # Track episode flags for sequence enforcement
        if parsed_action == "investigate":
            self._investigated = True

        if parsed_action == "assign":
            self._assigned = True

        mitigation_succeeded = (
            parsed_action == "mitigate"
            and keyword_score > 0
            and self._investigated
        )
        if mitigation_succeeded:
            self._mitigate_done = True

        severity_before = self._simulator.severity_score if self._simulator else None

        # 3. Advance the dynamic simulator so logs and metrics evolve per action.
        if self._simulator:
            sim_metrics, sim_logs = self._simulator.step(
                action_category=parsed_action,
                keyword_match_ratio=outcome.get("keyword_match_ratio", 0.0),
                mitigate_done=mitigation_succeeded,
                action_text=raw_text,
            )
            self._latest_metrics = sim_metrics
            self._latest_logs = sim_logs or self._latest_logs
        severity_after = self._simulator.severity_score if self._simulator else severity_before

        # Improvement streak
        if severity_after is not None and severity_before is not None:
            if severity_after < severity_before:
                self.improvement_streak += 1
            else:
                self.improvement_streak = 0

        # 4. Reward  (simulator-aware shaping + keyword grounding)
        step_reward = self._compute_step_reward(
            parsed_action,
            outcome,
            mitigation_succeeded=mitigation_succeeded,
            severity_before=severity_before,
            severity_after=severity_after,
        )
        base_step_reward = step_reward

        # Calibration reward: R2 (cross-entropy) + stability (KL penalty)
        # step_r2_reward in [0.0, 1.0] → scaled by R2_WEIGHT
        # step_stability_reward in [-0.10, 0.0] → applied directly
        step_reward += R2_WEIGHT * step_r2_reward
        step_reward += step_stability_reward
        step_reward  = max(-1.0, min(2.0, step_reward))

        # 5. Bookkeeping
        self._step_count += 1

        if parsed_action == "resolve" and self._can_resolve():
            self._resolved = True
            step_reward = max(-1.0, min(2.0, step_reward + TERMINAL_SUCCESS_BONUS))
            _debug(
                "terminal "
                f"task={self._task_id} step={self._step_count} "
                f"bonus={TERMINAL_SUCCESS_BONUS:.2f} reward={step_reward:.4f}"
            )

        # Better feedback so model knows what to do next
        if parsed_action == "mitigate" and mitigation_succeeded:
            if self._can_resolve():
                feedback = (
                    f"Fix applied: {matched}. LIVE METRICS are now healthy across the affected services. "
                    "You may resolve."
                )
            else:
                feedback = (
                    f"Fix applied: {matched}. LIVE METRICS are improving, but the affected services are still "
                    "degraded. Re-apply the same mitigation until the affected services are healthy before resolving."
                )
        elif parsed_action == "resolve" and self._resolved:
            feedback = "Incident resolved successfully."
        elif parsed_action == "resolve" and self._mitigate_done:
            feedback = (
                "Cannot resolve yet — mitigation has been applied, but LIVE METRICS show the affected "
                "services are still degraded. Re-apply the successful mitigation until the affected services are healthy."
            )
        elif parsed_action == "resolve" and not self._mitigate_done:
            feedback = "Cannot resolve — no mitigation applied yet. Investigate and mitigate first."
        elif parsed_action == "investigate":
            feedback = "Investigation complete. Assign a team or apply a mitigation."
        elif assigned_team_valid:
            feedback = "Team assigned. Now apply a mitigation."
        elif parsed_action == "assign":
            if any(team in action_lower for team in available_teams):
                feedback = "Team assigned, but it is not the correct owner for this incident. Use the exact team from the roster that matches the failing service."
            else:
                feedback = "Team name not recognized. Use an exact team name from the roster."
        elif parsed_action == "mitigate" and keyword_score > 0 and not self._investigated:
            feedback = "A plausible fix was attempted, but you must investigate the affected service before applying mitigation."
        elif parsed_action == "mitigate":
            feedback = "Mitigation failed. Reassess the logs and LIVE METRICS before trying another fix."
        else:
            feedback = "No matching keywords found."

        # Add final reward to cumulative (SLA decay is already in _compute_step_reward)
        self._cumulative_reward += step_reward
        _debug(
            "reward "
            f"task={self._task_id} step={self._step_count} "
            f"action={parsed_action} raw_action={action.action!r} "
            f"base={base_step_reward:.4f} r2_add={(R2_WEIGHT * step_r2_reward):.4f} "
            f"stability={step_stability_reward:.4f} final={step_reward:.4f} "
            f"cumulative={self._cumulative_reward:.4f} "
            f"mitigation_succeeded={mitigation_succeeded} can_resolve={self._can_resolve()}"
        )

        # --- Termination logic ---
        task_config = TASK_CONFIGS.get(self._task_id, {})
        max_steps = task_config.get("max_steps", 10)

        # Increment invalid counter BEFORE computing done
        if parsed_action == "resolve" and not self._can_resolve():
            self._consecutive_invalid += 1
        else:
            self._consecutive_invalid = 0

        done = (
            self._resolved
            or self._step_count >= max_steps
            or self._consecutive_invalid >= 3
        )
        if (
            parsed_action == "resolve"
            and severity_after is not None
            and severity_after > 0.10
        ):
            done = False

        # 7. Build observation and return the Observation expected by OpenEnv
        step_info = {
            "parsed_action": parsed_action,
            "outcome": outcome,
            "cumulative_reward": self._cumulative_reward,
            "keyword_score": keyword_score,
            "simulator_recovered": self._can_resolve(),
            "severity_before": severity_before,
            "severity_after": severity_after,
            "severity_delta": (
                None
                if severity_before is None or severity_after is None
                else severity_before - severity_after
            ),
            "root_cause_identified": self.root_cause_identified,
            "mitigation_attempts": self.mitigation_attempts,
            "improvement_streak": self.improvement_streak,
        }
        observation = self._build_observation(
            feedback=feedback,
            score=keyword_score,
            outcome=outcome,
            reward=step_reward,
            done=done,
            metadata=step_info,
            logs_override=self._latest_logs,
            metrics_override=self._latest_metrics,
        )
        return observation

    def state(self) -> IncidentState:
        """Return current environment state.  Must be a method, not a property."""
        # Expose incident metadata for grading via /state endpoint
        info: Dict[str, Any] = {}
        if self._incident_data:
            for key in ("root_cause", "team", "correct_team",
                        "root_cause_service", "affected_services",
                        "severity", "sla_minutes", "valid_mitigations"):
                if key in self._incident_data:
                    info[key] = self._incident_data[key]
            # Expose agent reasoning + runbook queries for grading
            info["agent_reasoning"] = " ".join(self._reasoning_texts)
            info["runbook_queries"] = list(self._runbook_queries)
            # Belief trajectory for R2 grader
            info["belief_trajectory"] = self._belief_trajectory
            info["last_reported_belief"] = self._last_reported_belief
            info["fan_in_candidates"] = self._incident_data.get("fan_in_candidates", [])
            info["hypotheses"] = self._incident_data.get("hypotheses", [])
            info["red_herring_services"] = self._incident_data.get("red_herring_services", [])
            # Agent-visible topology (nodes + noisy edges only)
            info["fan_in_dag"] = self._incident_data.get("fan_in_dag", {})
            # Grader-only ground truth (root, true/spurious/missing edges)
            info["fan_in_dag_ground_truth"] = self._incident_data.get("_fan_in_dag_ground_truth", {})
            # R2 calibration score over full belief trajectory
            info["r2_score"] = self._compute_r2_reward()
            _state_root = (
                self._incident_data.get("root_cause_service")
                or self._incident_data.get("root_cause", "")
            )
            _state_cands = self._belief_candidates()
            info["belief_xent_per_step"] = [
                self._compute_step_xent(b, _state_root, _state_cands)
                for b in self._belief_trajectory
            ]
            if self._simulator:
                info["simulator_snapshot"] = self._simulator.snapshot()
                info["simulator_recovered"] = self._simulator.is_fully_recovered()

        return IncidentState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            cumulative_reward=self._cumulative_reward,
            resolved=self._resolved,
            seed=self._seed,
            info=info,
        )

    @property
    def incident_data(self) -> Dict[str, Any]:
        """Expose incident data for the /incident-meta endpoint."""
        return self._incident_data

    # ------------------------------------------------------------------
    # Belief parsing  (agent self-reports belief each step)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_belief_from_reasoning(reasoning: str) -> Optional[Dict[str, float]]:
        """Extract belief dict from agent reasoning text.

        Looks for pattern: Belief: {"auth-service": 0.7, "db-primary": 0.2}
        Returns None on any parse failure — caller assigns R2=0, does NOT drop rollout.
        """
        if not reasoning:
            return None
        import ast
        match = re.search(r'[Bb]elief\s*[:=]\s*(\{[^}]+\})', reasoning)
        if not match:
            return None
        try:
            raw = match.group(1)
            # Quote only UNQUOTED hyphenated keys.
            # Old regex `([\w][\w\-]*)\s*:` re-wrapped already-quoted keys,
            # producing ""auth-service"": which ast.literal_eval rejects silently.
            raw = re.sub(r'(?<!")(\b[\w][\w\-]*\b)(?!")\s*:', r'"\1":', raw)
            parsed = ast.literal_eval(raw)
            if not isinstance(parsed, dict):
                return None
            # Normalise to float values
            return {str(k): float(v) for k, v in parsed.items()}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # R2 calibration reward — cross-entropy belief trajectory grading
    # ------------------------------------------------------------------

    def _belief_candidates(self) -> List[str]:
        """Return the active candidate set for R2 grading."""
        return (
            self._incident_data.get("fan_in_candidates")
            or self._incident_data.get("hypotheses")
            or self._incident_data.get("affected_services", [])
        )

    @staticmethod
    def _normalize_belief(
        belief: Dict[str, float],
        candidates: List[str],
    ) -> Dict[str, float]:
        """Project belief onto candidates and normalise to a distribution."""
        if not candidates:
            return dict(belief)

        projected: Dict[str, float] = {}
        for key in candidates:
            try:
                projected[key] = max(0.0, float(belief.get(key, 0.0)))
            except (TypeError, ValueError):
                projected[key] = 0.0

        total = sum(projected.values())
        if total > 0.0:
            return {key: value / total for key, value in projected.items()}
        return projected

    @staticmethod
    def _compute_step_xent(
        belief: Dict[str, float],
        root: str,
        candidates: List[str],
    ) -> float:
        """Cross-entropy calibration reward for one belief snapshot.

        Implements: −Σ_s p(s) log p̂(s)  with one-hot ground truth p.
        Since p(root) = 1.0 and p(other) = 0.0, this simplifies to:

            CE_t = −log(p̂_t(root))

        Normalised by log(N) (N = |candidates|) so that:
            p̂(root) = 1.0  →  reward = 1.0   (perfect)
            p̂(root) = 1/N  →  reward = 0.0   (uniform, baseline)
            p̂(root) < 1/N  →  reward = 0.0   (clamped; worse than uniform)

        Returns a value in [0.0, 1.0].
        """
        import math
        N = len(candidates)
        if N <= 1:
            return 1.0   # trivially correct — no ambiguity to resolve
        belief = IncidentResponseEnv._normalize_belief(belief, candidates)
        eps = 1e-9
        p_root = max(float(belief.get(root, 0.0)), eps)
        # Normalise: 1 + log(p_root)/log(N)
        #   log(1.0)/log(N)  = 0    → reward = 1.0
        #   log(1/N)/log(N)  = -1   → reward = 0.0
        reward = 1.0 + math.log(p_root) / math.log(N)
        return max(0.0, min(1.0, reward))

    def _compute_r2_reward(self) -> float:
        """Episode R2: cross-entropy calibration averaged over ALL episode steps.

        Enabled for ALL tasks with more than one candidate service to
        discriminate between (i.e. N > 1).  This covers:
          cascading_failure         → fan_in_candidates (varied per seed)
          ambiguous_payment_degradation → affected_services (3 services)
          single_service_outage     → N=1, returns 0.0 (no ambiguity)

        Root resolution (first non-empty wins):
          root_cause_service  → set by cascading_failure / task 1
          root_cause          → set by ambiguous_payment_degradation task 3

        Candidates resolution (first non-empty wins):
          fan_in_candidates   → set by build_fan_in_dag
          affected_services   → universal fallback

        Formula: R2 = (1/T) * Σ_{t=1}^{T} xent_t
        Zero-scores missing belief steps (no rollout drop).
        Range: [0.0, 1.0]
        """
        root = (
            self._incident_data.get("root_cause_service")
            or self._incident_data.get("root_cause", "")
        )
        candidates = self._belief_candidates()

        # R2 only fires when there is genuine ambiguity (N > 1)
        if not root or len(candidates) <= 1:
            return 0.0
        if not self._belief_trajectory:
            return 0.0

        total_steps = max(self._step_count, 1)
        xent_sum = sum(
            self._compute_step_xent(b, root, candidates)
            for b in self._belief_trajectory
        )
        return max(0.0, min(1.0, xent_sum / total_steps))

    def _compute_stability_reward(self) -> float:
        """Symmetric KL-divergence penalty for oscillating belief.

        Compares the last two entries in self._belief_trajectory.
        Call AFTER appending the new belief so index [-1] is current,
        index [-2] is previous.

        Returns a value in [-0.10, 0.0]:
            0.0   — belief is stable (low symmetric KL)
            -0.10 — belief oscillates wildly (high symmetric KL)

        Replaces explicit belief injection in the prompt with an *incentive*
        for smooth, evidence-driven updates instead of random flipping.
        """
        if len(self._belief_trajectory) < 2:
            return 0.0
        import math
        prev = self._belief_trajectory[-2]
        curr = self._belief_trajectory[-1]
        all_keys = set(prev.keys()) | set(curr.keys())
        eps = 1e-9
        sym_kl = 0.0
        for k in all_keys:
            p = max(float(prev.get(k, 0.0)), eps)
            q = max(float(curr.get(k, 0.0)), eps)
            # Symmetric KL: KL(p‖q) + KL(q‖p)
            sym_kl += p * math.log(p / q) + q * math.log(q / p)
        # Map to [-0.10, 0.0]: sym_kl ≈ 0 → 0 penalty; sym_kl ≥ 2 → -0.10
        penalty = min(0.10, sym_kl * 0.05)
        return -penalty

    # ------------------------------------------------------------------
    # Action parsing  (pure keyword matching)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(text: str) -> str:
        """Parse free-form action text into a canonical category.

        Returns one of:
            ``investigate | assign | escalate | mitigate | resolve | unknown``
        """
        text_lower = text.lower()
        for category, keywords in _ACTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return category
        return "unknown"

    # ------------------------------------------------------------------
    # Outcome computation  (pure Python, no LLM)
    # ------------------------------------------------------------------

    def _compute_outcome(self, parsed_action: str) -> Dict[str, Any]:
        """Derive an outcome dict from the parsed action category.

        Uses ``self._incident_data`` (the generated incident) as the
        single source of truth for grading criteria.
        """
        valid_mitigations: List[str] = self._incident_data.get("valid_mitigations", [])
        severity: str = self._incident_data.get("severity", "P2")

        # Severity weight mapping
        severity_weights = {"P0": 1.0, "P1": 0.8, "P2": 0.5}
        severity_weight = severity_weights.get(severity, 0.5)

        # Keyword match ratio against last raw action
        if valid_mitigations and self._last_action:
            action_lower = self._last_action.lower()
            matched = sum(1 for kw in valid_mitigations if kw.lower() in action_lower)
            keyword_match_ratio = matched / len(valid_mitigations)
        else:
            keyword_match_ratio = 0.0

        # Redundancy check
        redundant = self._is_redundant(parsed_action)

        return {
            "action_valid": parsed_action != "unknown",
            "action_category": parsed_action,
            "severity_weight": severity_weight,
            "keyword_match_ratio": keyword_match_ratio,
            "redundant_action": redundant,
        }

    def _is_redundant(self, parsed_action: str) -> bool:
        """True only when this specific action adds no new information."""
        history = self._action_history[:-1]  # exclude current action
        if not history:
            return False

        raw_lower = (self._last_action or "").lower()

        if parsed_action == "investigate":
            # Only redundant if the SAME service was already investigated
            affected = self._incident_data.get("affected_services", [])
            for svc in affected:
                if re.search(r'(?<![a-z0-9-])' + re.escape(svc.lower()) + r'(?![a-z0-9-])', raw_lower):
                    # Already investigated this exact service
                    past_invs = [a for a in history
                                 if self._parse_action(a.lower()) == "investigate"
                                 and re.search(r'(?<![a-z0-9-])' + re.escape(svc.lower()) + r'(?![a-z0-9-])', a.lower())]
                    if past_invs:
                        return True
            return False

        if parsed_action in ("resolve", "escalate"):
            # Redundant if already done once
            return any(self._parse_action(a.lower()) == parsed_action for a in history)

        if parsed_action == "mitigate":
            # After full recovery, any repeat is redundant.
            if self._mitigate_done and self._can_resolve():
                return True

            # Before recovery, penalise exact-duplicate text after the 2nd occurrence.
            raw_lower = (self._last_action or "").lower().strip()

            exact_repeats = sum(
                1 for a in history
                if self._parse_action(a.lower()) == "mitigate"
                and a.lower().strip() == raw_lower
            )

            return exact_repeats >= 2

        return False

    # ------------------------------------------------------------------
    # Reward computation  (pure arithmetic – NO LLM)
    # ------------------------------------------------------------------

    def _compute_step_reward(
        self,
        parsed_action: str,
        outcome: Dict[str, Any],
        mitigation_succeeded: bool = False,
        severity_before: Optional[float] = None,
        severity_after: Optional[float] = None,
    ) -> float:
        """Compute a single-step scalar reward from the outcome dict.

        Reward is grounded in two signals:
        - structural correctness (investigate the right services, assign the right team)
        - simulator impact (did the action actually improve live system health?)

        Returns:
            A float reward, clamped to [-1.0, 2.0].
        """
        keyword_ratio: float = outcome.get("keyword_match_ratio", 0.0)
        redundant: bool = outcome.get("redundant_action", False)

        reward = 0.0
        current_step = self._step_count + 1
        severity_delta = 0.0
        if severity_before is not None and severity_after is not None:
            severity_delta = severity_before - severity_after

        # --- Base reward by action type ---
        if parsed_action == "investigate":
            investigations_done = sum(
                1 for a in self._action_history[:-1]
                if self._parse_action(a.lower()) == "investigate"
            )
            # Diminishing returns: 0.15, 0.12, 0.09, 0.06, 0.03 (floor 0.02)
            base = max(0.02, 0.15 - (investigations_done * 0.03))

            # Check if investigating an actually affected service
            services = self._incident_data.get("affected_services", [])
            last = (self._last_action or "").lower()
            
            # Use simple substring match — hyphenated names work fine with 'in'
            investigating_affected = any(svc.lower() in last for svc in services)
            
            if investigating_affected:
                reward = base + 0.10  # correct: 0.25 → 0.12 → ... 
            else:
                reward = -0.05  # wrong service: flat small penalty, not -0.10

        elif parsed_action == "assign":
            correct_team = self._incident_data.get("correct_team", "").lower()
            available_teams = {
                team.lower() for team in self._incident_data.get("team_map", {}).values()
            }
            last = (self._last_action or "").lower()
            if correct_team and correct_team in last:
                reward = 0.10
            elif any(team in last for team in available_teams):
                reward = 0.02
            else:
                reward = -0.05

        elif parsed_action == "escalate":
            reward = 0.08 if self._step_count <= 3 else 0.01

        elif parsed_action == "mitigate":
            if not self._investigated:
                reward -= 0.30
            elif not mitigation_succeeded:
                reward -= 0.15
            else:
                reward += 0.03 + (keyword_ratio * 0.10)  # base reduced 0.05->0.03 (R2_WEIGHT raised)

            if self.root_cause_identified:
                reward += 0.20

            if self.mitigation_attempts > 2:
                reward -= 0.10 * (self.mitigation_attempts - 2)

            if severity_before is not None and severity_after is not None:
                if severity_after < severity_before:
                    reward += severity_delta
                if severity_after >= severity_before:
                    reward -= 0.50
                elif severity_delta < 0.03:
                    reward -= 0.20

            if self.improvement_streak >= 2:
                reward += 0.20

        elif parsed_action == "resolve":
            current_severity = severity_after if severity_after is not None else 1.0

            if not self._can_resolve():
                reward -= 0.60  # premature resolve = heavy penalty

            elif current_severity > 0.05:
                reward -= 0.40  # system not fully healthy
            else:
                reward += max(0.3, 0.8 - (0.08 * current_step))  # reduced reward

        else:
            reward -= 0.30

        if severity_before is not None and severity_after is not None:
            if parsed_action in ("investigate", "assign", "escalate") and severity_delta > 0.05:
                reward += 0.05
            elif parsed_action == "unknown" and severity_delta < -0.05:
                reward -= 0.10

        # Task 3 exploration penalty — applied ONCE on the final step only.
        # The old per-step version applied -0.40 EVERY step until all 3 hypotheses
        # were explored.  Over 8 steps with 1 hypothesis: 8 * -0.40 = -3.20.
        # This made cumulative reward catastrophically negative (R=-6.28) and
        # overwhelmed any positive signal from correct investigation actions.
        if (self._task_id == "ambiguous_payment_degradation"
                and (self._step_count + 1 >= 8 or parsed_action == "resolve")):
            total_hypotheses = 3
            explored = 0
            if any("db" in a.lower() for a in self._action_history):
                explored += 1
            if any("rate" in a.lower() for a in self._action_history):
                explored += 1
            if any("memory" in a.lower() for a in self._action_history):
                explored += 1
            remaining = max(0, total_hypotheses - explored)
            reward -= 0.40 * (remaining / total_hypotheses)

        # Stronger redundancy penalty
        if redundant:
            reward -= 0.50

        # SLA pressure — penalty increases as deadline approaches
        sla_minutes = self._incident_data.get("sla_minutes", 30)
        time_used = current_step * 2
        sla_fraction = min(1.0, time_used / max(sla_minutes, 1))
        if sla_fraction > 0.7:
            reward -= (sla_fraction - 0.7) * 0.10

        reward -= STEP_COST

        return max(-1.0, min(2.0, reward))

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        feedback: str,
        score: float,
        outcome: Dict[str, Any],
        reward: float = 0.0,
        done: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        logs_override: Optional[List[str]] = None,
        metrics_override: Optional[List[Dict[str, Any]]] = None,
    ) -> IncidentObservation:
        """Construct a fully-populated observation from generated incident data.

        The returned IncidentObservation inherits from the framework's
        Observation base, so it carries reward, done, and metadata fields
        that the serializer reads directly.
        """
        if not self._initialized or not self._incident_data:
            raise RuntimeError(
                "Call reset() before building observations. "
                "IncidentResponseEnv is not initialized."
            )

        inc = self._incident_data
        task_config = TASK_CONFIGS.get(self._task_id, {})

        # Build description from generated incident
        description = inc.get("title", "")
        if inc.get("logs"):
            description += "\n\nRecent logs:\n" + "\n".join(inc["logs"][:3]) + "\n..."

        # SLA remaining in observation
        sla_minutes = inc.get("sla_minutes", 30)
        sla_remaining = max(0, sla_minutes - (self._step_count * 2))

        from models import ServiceMetrics
        logs = logs_override if logs_override is not None else self._latest_logs
        metrics_source = (
            metrics_override if metrics_override is not None else self._latest_metrics
        )
        metrics = [ServiceMetrics(**metric) for metric in metrics_source]

        # Team roster — shows availability without giving away correct team
        team_map: Dict[str, str] = inc.get("team_map", {})
        team_roster = {team: "available" for team in set(team_map.values())}

        # Simulator-first progress score
        if self._simulator:
            normalized_cumulative = max(
                0.0,
                min(1.0, 1.0 - self._simulator.severity_score),
            )
        else:
            max_possible = task_config.get("max_reward", 1.0)
            normalized_cumulative = min(
                1.0, max(0.0, self._cumulative_reward / max(max_possible, 0.01))
            )

        return IncidentObservation(
            # Framework-required fields (serializer reads these)
            reward=reward,
            done=done,
            metadata=metadata or {},
            # Domain fields
            task_id=self._task_id,
            incident_description=description,
            affected_services=inc.get("affected_services", []),
            severity=inc.get("severity", "P2"),
            logs=logs[-8:],
            metrics=metrics,
            feedback=feedback,
            score=round(normalized_cumulative, 4),
            step_count=self._step_count,
            resolved=self._resolved,
            last_action=self._last_action,
            outcome=outcome,
            sla_remaining=sla_remaining,
            team_roster=team_roster,
            # Belief candidates for R2-aligned belief prompting
            hypotheses=inc.get("hypotheses", []),
            fan_in_candidates=inc.get("fan_in_candidates", []),
        )
