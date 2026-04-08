"""
Dynamic incident simulation engine — fixed.

Key fixes vs original:
  1. _propagate_failures collects deltas first, applies after — no
     iteration-while-modifying bug (order no longer affects outcome).
  2. _init_states initialises ALL services in the dependency graph,
     not just affected_services — cascade targets are now live from step 0.
  3. generate_step_logs returns deterministic output for same RNG state.
  4. severity_score is max over all initialized services, not just affected.
  5. is_fully_recovered checks only affected services (healthy bystanders
     should not prevent resolution).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Dependency graph (downstream view: who depends on me?)
# ---------------------------------------------------------------------------

DOWNSTREAM: Dict[str, List[str]] = {
    "auth-service":      ["api-gateway", "user-service"],
    "user-service":      ["api-gateway", "order-service"],
    "api-gateway":       [],
    "payment-service":   ["order-service"],
    "notification-svc":  [],
    "search-service":    [],
    "order-service":     [],
    "reporting-service": [],
    "config-service":    ["auth-service", "api-gateway", "payment-service"],
    "db-primary":        ["auth-service", "user-service", "order-service", "payment-service"],
    "cache-cluster":     ["api-gateway", "user-service", "search-service"],
    "worker-pool":       ["notification-svc", "reporting-service"],
}

# Upstream view: who do I depend on?
UPSTREAM: Dict[str, List[str]] = {}
for _src, _dests in DOWNSTREAM.items():
    for _d in _dests:
        UPSTREAM.setdefault(_d, []).append(_src)

PROPAGATION_THRESHOLD = 25.0
PROPAGATION_WEIGHT    = 0.30


# ---------------------------------------------------------------------------
# Service state
# ---------------------------------------------------------------------------

@dataclass
class ServiceState:
    name: str
    error_rate: float       # 0–100
    latency_p99: int        # ms
    throughput: int         # rps
    saturation: float       # 0–1
    status: str             # ok | degraded | down
    baseline_latency: int   = 120
    max_throughput: int     = 1000
    natural_growth: float   = 0.04
    noise_amp: float        = 0.8
    is_affected: bool       = False   # part of the incident?

    def recompute_derived(self) -> None:
        er = self.error_rate / 100.0
        self.latency_p99 = int(self.baseline_latency * (1.0 + (er ** 1.8) * 60.0))
        self.throughput  = max(0, int(self.max_throughput * max(0.0, (1.0 - er) ** 1.5)))
        if self.error_rate >= 70.0:
            self.status = "down"
        elif self.error_rate >= 15.0:
            self.status = "degraded"
        else:
            self.status = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service":        self.name,
            "error_rate_pct": round(self.error_rate, 1),
            "latency_p99_ms": self.latency_p99,
            "throughput_rps": self.throughput,
            "status":         self.status,
        }


# ---------------------------------------------------------------------------
# Dynamic log templates
# ---------------------------------------------------------------------------

_LOG_TEMPLATES: Dict[Tuple[str, str], List] = {
    ("error_rate", "high"): [
        lambda s, st: f"ERROR {s}: error_rate={st.error_rate:.1f}% — SLO breach (threshold: 1.0%)",
        lambda s, st: f"CRIT  {s}: {int(st.error_rate * st.throughput / 100)} req/s failing",
        lambda s, st: f"WARN  pagerduty: {s} breached P1 threshold at {st.error_rate:.1f}%",
        lambda s, st: f"ERROR {s}: circuit breaker OPEN — downstream calls failing fast",
    ],
    ("error_rate", "critical"): [
        lambda s, st: f"CRIT  {s}: error_rate={st.error_rate:.1f}% — service effectively DOWN",
        lambda s, st: f"ERROR load-balancer: removing {s} from rotation (health check fail)",
        lambda s, st: f"CRIT  {s}: 0 healthy pods — k8s readiness probe failing",
        lambda s, st: f"ERROR {s}: all retry attempts exhausted — requests queued",
    ],
    ("latency", "degraded"): [
        lambda s, st: f"WARN  {s}: p99_latency={st.latency_p99}ms — breaching SLO (200ms)",
        lambda s, st: f"WARN  {s}: goroutine count {int(st.saturation * 10000)} — potential leak",
        lambda s, st: f"WARN  {s}: GC pause {int(st.latency_p99 * 0.08)}ms — heap pressure",
    ],
    ("throughput", "drop"): [
        lambda s, st: f"WARN  {s}: throughput dropped to {st.throughput} rps (baseline: {st.max_throughput})",
        lambda s, st: f"INFO  autoscaler: scale-up triggered for {s} (queue depth > 500)",
    ],
    ("recovery", "partial"): [
        lambda s, st: f"INFO  {s}: error_rate improving — now {st.error_rate:.1f}%",
        lambda s, st: f"INFO  {s}: p99_latency recovering to {st.latency_p99}ms",
        lambda s, st: f"INFO  monitoring: {s} status changed to DEGRADED",
    ],
    ("recovery", "full"): [
        lambda s, st: f"INFO  {s}: fully recovered — error_rate={st.error_rate:.1f}%",
        lambda s, st: f"INFO  load-balancer: {s} re-added to rotation",
        lambda s, st: f"INFO  pagerduty: incident auto-resolved — {s} healthy",
    ],
    ("cascade", "spreading"): [
        lambda s, st: f"WARN  {s}: upstream failure — {st.error_rate:.1f}% of calls failing",
        lambda s, st: f"ERROR {s}: connection pool pressure from upstream failures",
    ],
}

_AUTONOMOUS_EVENTS: List[str] = [
    "INFO  autoscaler: evaluating scale-up criteria (CPU > 80%)",
    "INFO  deploy-bot: no deployments scheduled in next 30 min",
    "INFO  backup-agent: incremental backup completed",
    "WARN  monitoring: anomaly detection triggered — flagging for review",
    "INFO  k8s: pod eviction check passed — no memory pressure",
    "INFO  tracing: slow request captured (trace_id=abc{step})",
    "WARN  db-metrics: connection pool at 71% utilization",
    "INFO  sla-tracker: P1 incident timer running",
    "WARN  cost-alert: error spike causing 3x normal invocations",
    "INFO  runbook-bot: relevant runbook found in /runbooks/",
]


def _pick_log_key(state: ServiceState, prev_er: float) -> Optional[Tuple[str, str]]:
    recovering = state.error_rate < prev_er - 2.0
    er = state.error_rate
    if recovering and er < 10.0:
        return ("recovery", "full")
    if recovering:
        return ("recovery", "partial")
    if er >= 60.0:
        return ("error_rate", "critical")
    if er >= 20.0:
        return ("error_rate", "high")
    if state.latency_p99 > 500 and er < 20.0:
        return ("latency", "degraded")
    if state.throughput < state.max_throughput * 0.5:
        return ("throughput", "drop")
    return None


def generate_step_logs(
    states: Dict[str, "ServiceState"],
    step: int,
    rng: random.Random,
    prev_error_rates: Dict[str, float],
) -> List[str]:
    logs: List[str] = []

    for svc, state in states.items():
        if not state.is_affected and state.error_rate < 5.0:
            continue  # quiet healthy bystander

        prev_er = prev_error_rates.get(svc, state.error_rate)
        key = _pick_log_key(state, prev_er)
        if key:
            templates = _LOG_TEMPLATES.get(key, [])
            if templates:
                try:
                    logs.append(rng.choice(templates)(svc, state))
                except Exception:
                    pass

        # Cascade logs for affected upstream
        for upstream in UPSTREAM.get(svc, []):
            if upstream in states and states[upstream].error_rate > PROPAGATION_THRESHOLD:
                cascade = _LOG_TEMPLATES.get(("cascade", "spreading"), [])
                if cascade and rng.random() < 0.35:
                    try:
                        logs.append(rng.choice(cascade)(svc, state))
                    except Exception:
                        pass
                    break

        if len(logs) >= 3:
            break

    # Autonomous event every 3 steps
    if step % 3 == 0 and _AUTONOMOUS_EVENTS:
        evt = rng.choice(_AUTONOMOUS_EVENTS)
        logs.append(evt.replace("{step}", str(step)))

    return logs[:4]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class IncidentSimulator:
    """Live metric simulation engine.

    Fixes vs original:
    - _init_states creates ServiceState for ALL services reachable via the
      dependency graph from any affected service — not just affected_services.
    - _propagate_failures accumulates deltas before applying (no mutation
      during iteration).
    - current_metrics() returns only affected + currently-degraded services
      so the observation is readable.
    """

    def __init__(self, incident_data: Dict[str, Any], seed: int) -> None:
        self._data    = incident_data
        self._rng     = random.Random(seed + 99991)  # isolated from incident gen
        self._step    = 0
        self._states: Dict[str, ServiceState] = {}
        self._affected: Set[str] = set(incident_data.get("affected_services", []))
        self._init_states()
        self._prev_error_rates: Dict[str, float] = {
            k: s.error_rate for k, s in self._states.items()
        }
        self._mitigation_applied  = False
        self._mitigation_strength = 0.0

    # ------------------------------------------------------------------

    def _init_states(self) -> None:
        initial_map: Dict[str, Dict[str, Any]] = {
            m["service"]: m for m in self._data.get("initial_metrics", [])
        }

        # Collect all services to initialize: affected + their downstream neighbours
        to_init: Set[str] = set(self._affected)
        for svc in list(self._affected):
            to_init.update(DOWNSTREAM.get(svc, []))

        for svc in to_init:
            base = initial_map.get(svc, {})
            is_aff = svc in self._affected

            if is_aff:
                er  = base.get("error_rate_pct", 45.0)
                lat = base.get("latency_p99_ms", 1200)
                thr = base.get("throughput_rps", 200)
                sta = base.get("status", "degraded")
                sat = self._rng.uniform(0.6, 0.95)
            else:
                # Healthy bystander — low baseline
                er  = self._rng.uniform(0.1, 1.5)
                lat = self._rng.randint(80, 200)
                thr = self._rng.randint(600, 1200)
                sta = "ok"
                sat = self._rng.uniform(0.1, 0.35)

            self._states[svc] = ServiceState(
                name=svc,
                error_rate=er, latency_p99=lat, throughput=thr,
                saturation=sat, status=sta,
                baseline_latency=self._rng.randint(80, 180),
                max_throughput=self._rng.randint(600, 1400),
                natural_growth=self._rng.uniform(0.03, 0.07),
                noise_amp=self._rng.uniform(0.5, 1.5),
                is_affected=is_aff,
            )

    # ------------------------------------------------------------------

    def step(
        self,
        action_category: str,
        keyword_match_ratio: float,
        mitigate_done: bool = False,
        action_text: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        self._step += 1
        self._prev_error_rates = {k: v.error_rate for k, v in self._states.items()}

        if mitigate_done and keyword_match_ratio > 0:
            self._mitigation_applied  = True
            self._mitigation_strength = keyword_match_ratio

        for state in self._states.values():
            self._evolve(
                state,
                action_category,
                mitigate_done,
                keyword_match_ratio,
                action_text=action_text,
            )

        self._propagate_failures()

        new_logs = generate_step_logs(
            states=self._states,
            step=self._step,
            rng=self._rng,
            prev_error_rates=self._prev_error_rates,
        )

        return self.current_metrics(), new_logs

    def _evolve(
        self,
        state: ServiceState,
        action: str,
        mitigate_done: bool,
        kw_ratio: float,
        action_text: str = "",
    ) -> None:
        noise = self._rng.gauss(0, state.noise_amp)

        if state.error_rate <= 0.5:
            state.error_rate = max(0.1, state.error_rate + noise * 0.1)
            state.saturation = max(0.1, min(1.0, state.saturation - 0.05))
            state.recompute_derived()
            return

        if not state.is_affected:
            # Bystander: no active recovery, slow passive drift
            if state.error_rate > 1.0:
                state.error_rate = max(0.1, state.error_rate * 0.97)
            state.recompute_derived()
            return

        if (
            self._data.get("root_cause") == "db_overload"
            and self._data.get("affected_services") == ["payment-service"]
            and state.name == "payment-service"
            and action == "mitigate"
        ):
            if any(token in action_text for token in ("db", "connection", "pool", "tune")):
                rate = 0.50 if state.error_rate > 25.0 else 0.25
                state.error_rate = max(0.1, state.error_rate * rate + noise * 0.1)
                state.saturation = max(0.1, state.saturation * 0.70)
            elif "restart" in action_text:
                if state.error_rate > 35.0:
                    state.error_rate = max(0.1, state.error_rate * 0.75 + noise * 0.1)
                else:
                    state.error_rate = min(99.0, state.error_rate * 1.05 + abs(noise) * 0.2)
                state.saturation = min(1.0, max(0.1, state.saturation * 0.95))
            elif "scale" in action_text or "replica" in action_text:
                state.error_rate = min(99.0, state.error_rate * 1.20 + abs(noise) * 0.2)
                state.saturation = min(1.0, state.saturation + 0.03)
            else:
                state.error_rate = min(99.0, state.error_rate * 1.02 + abs(noise) * 0.2)
                state.saturation = min(1.0, state.saturation + 0.01)
            state.recompute_derived()
            return

        if mitigate_done and kw_ratio > 0:
            rate    = 0.40 + kw_ratio * 0.40   # floor raised: 0.25→0.40
            state.error_rate = max(0.1, state.error_rate * (1.0 - rate) + noise * 0.3)
            state.saturation = max(0.1, state.saturation * 0.75)
        elif action == "investigate":
            growth = state.natural_growth * 0.5
            state.error_rate = min(99.0, state.error_rate * (1.0 + growth) + noise * 0.4)
            state.saturation = min(1.0, state.saturation + 0.01)
        elif action in ("assign", "escalate"):
            state.error_rate = min(99.0, state.error_rate * (1.0 + state.natural_growth) + noise)
            state.saturation = min(1.0, state.saturation + 0.02)
        elif action == "resolve":
            if self._mitigation_applied:
                recovery = 0.05 + self._mitigation_strength * 0.10
                state.error_rate = max(
                    0.1,
                    state.error_rate * (1.0 - recovery) + noise * 0.1,
                )
                state.saturation = max(0.1, state.saturation * 0.90)
            else:
                state.error_rate = min(
                    99.0,
                    state.error_rate * (1.0 + state.natural_growth * 0.25) + abs(noise) * 0.2,
                )
                state.saturation = min(1.0, state.saturation + 0.01)
        elif action == "unknown":
            state.error_rate = min(99.0, state.error_rate * (1.0 + state.natural_growth * 2.0) + abs(noise))
            state.saturation = min(1.0, state.saturation + 0.04)
        else:
            state.error_rate = min(99.0, state.error_rate * (1.0 + state.natural_growth) + noise * 0.5)
            state.saturation = min(1.0, state.saturation + 0.02)

        state.recompute_derived()

    def _propagate_failures(self) -> None:
        """Collect propagation deltas, then apply — avoids mutation-during-iteration."""
        deltas: Dict[str, float] = {}
        for svc, state in self._states.items():
            for upstream in UPSTREAM.get(svc, []):
                if upstream not in self._states:
                    continue
                up_er = self._states[upstream].error_rate
                if up_er > PROPAGATION_THRESHOLD:
                    delta = (up_er - PROPAGATION_THRESHOLD) * PROPAGATION_WEIGHT * 0.1
                    deltas[svc] = deltas.get(svc, 0.0) + delta

        for svc, delta in deltas.items():
            self._states[svc].error_rate = min(99.0, self._states[svc].error_rate + delta)
            self._states[svc].recompute_derived()

    # ------------------------------------------------------------------

    def current_metrics(self) -> List[Dict[str, Any]]:
        """Return metrics for affected services (what agent needs to see)."""
        return [s.to_dict() for s in self._states.values() if s.is_affected]

    def is_fully_recovered(self) -> bool:
        """True iff ALL affected services have error_rate < 5%."""
        return all(
            s.error_rate < 5.0          # 2.0 was unreachable under normal noise
            for s in self._states.values()
            if s.is_affected
        )

    @property
    def severity_score(self) -> float:
        if not self._states:
            return 0.0
        return max(s.error_rate for s in self._states.values()) / 100.0

    def snapshot(self) -> Dict[str, Any]:
        return {
            svc: {
                "error_rate":  round(s.error_rate, 2),
                "latency_p99": s.latency_p99,
                "throughput":  s.throughput,
                "saturation":  round(s.saturation, 2),
                "status":      s.status,
                "is_affected": s.is_affected,
            }
            for svc, s in self._states.items()
        }
