"""
Seeded incident generator for the Incident Response Environment.

Each incident is a plain dict containing the scenario data.  No LLM calls are
used – all content is static and deterministic.

The main entry point is ``generate_incident(task_id, seed)`` which produces a
fully-populated incident dict whose random elements (root-cause selection, log
ordering, timing jitter) are deterministic for a given seed.

Design rules:
    - ``rng = random.Random(seed)`` is created ONCE at the top of
      ``generate_incident``.
    - Inner generators do NOT create or re-seed their own RNG.
    - All ``rng.*`` calls are pre-generated before building data structures
      so adding/removing a log line never shifts the seed sequence.
"""

import random
from datetime import datetime, timezone
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Timestamp helper — makes logs look like real production output
# ---------------------------------------------------------------------------

def _format_ts(base_epoch: int, offset_minutes: int) -> str:
    """Return an ISO-8601 UTC timestamp string.

    Deterministic given *base_epoch* and *offset_minutes*.
    """
    t = datetime.fromtimestamp(
        base_epoch + offset_minutes * 60, tz=timezone.utc
    )
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp_logs(logs: List[str], base_epoch: int,
                interval_minutes: int = 2) -> List[str]:
    """Prepend a synthetic timestamp to each log line."""
    return [
        f"{_format_ts(base_epoch, i * interval_minutes)} {line}"
        for i, line in enumerate(logs)
    ]


# ---------------------------------------------------------------------------
# Valid task IDs  (single source of truth)
# ---------------------------------------------------------------------------

VALID_TASK_IDS = {
    "single_service_outage",
    "cascading_failure",
    "ambiguous_payment_degradation",
}


_SINGLE_SERVICE_BLUEPRINTS: List[Dict[str, Any]] = [
    {
        "kind": "connection_pool",
        "service": "auth-service",
        "title": "Auth service unavailable",
        "severity": "P1",
        "correct_team": "backend",
        "team_map": {
            "auth-service": "backend",
            "db-primary": "database",
        },
        "valid_mitigations": ["restart", "pool", "connection"],
    },
    {
        "kind": "feature_flag",
        "service": "api-gateway",
        "title": "API gateway request failures",
        "severity": "P1",
        "correct_team": "platform",
        "team_map": {
            "api-gateway": "platform",
            "config-service": "platform",
            "auth-service": "backend",
        },
        "valid_mitigations": ["rollback", "revert", "feature flag", "disable"],
    },
    {
        "kind": "memory_leak",
        "service": "payment-service",
        "title": "Payment pods crash looping",
        "severity": "P1",
        "correct_team": "payments-oncall",
        "team_map": {
            "payment-service": "payments-oncall",
            "stripe-api": "payments-oncall",
            "db-primary": "database",
        },
        "valid_mitigations": ["restart", "rollback", "memory", "heap"],
    },
    {
        "kind": "db_saturation",
        "service": "db-primary",
        "title": "Database saturation causing request failures",
        "severity": "P1",
        "correct_team": "database",
        "team_map": {
            "db-primary": "database",
            "db-replica": "database",
            "api-gateway": "platform",
        },
        "valid_mitigations": ["scale", "connections", "pool", "kill query"],
    },
    {
        "kind": "index_corruption",
        "service": "search-service",
        "title": "Search queries timing out",
        "severity": "P2",
        "correct_team": "search",
        "team_map": {
            "search-service": "search",
            "indexer": "search",
            "api-gateway": "platform",
        },
        "valid_mitigations": ["rebuild", "index", "purge", "cache"],
    },
    {
        "kind": "canary_regression",
        "service": "inventory-service",
        "title": "Inventory API regression after canary deploy",
        "severity": "P1",
        "correct_team": "backend",
        "team_map": {
            "inventory-service": "backend",
            "checkout-service": "backend",
            "api-gateway": "platform",
        },
        "valid_mitigations": ["rollback", "revert", "disable", "canary"],
    },
]


_CASCADE_BLUEPRINTS: List[Dict[str, Any]] = [
    {
        "kind": "feature_flag_rollout",
        "title": "Cascading failure after gateway config rollout",
        "severity": "P0",
        "affected_services": ["api-gateway", "user-service", "db-replica"],
        "root_cause_service": "api-gateway",
        "correct_team": "platform",
        "team_map": {
            "api-gateway": "platform",
            "user-service": "backend",
            "db-replica": "database",
        },
        "valid_mitigations": ["rollback", "revert", "feature flag", "disable"],
    },
    {
        "kind": "auth_config_regression",
        "title": "Authentication rollout cascading across user traffic",
        "severity": "P0",
        "affected_services": ["auth-service", "api-gateway", "user-service"],
        "root_cause_service": "auth-service",
        "correct_team": "backend",
        "team_map": {
            "auth-service": "backend",
            "api-gateway": "platform",
            "user-service": "backend",
        },
        "valid_mitigations": ["rollback", "revert", "config", "disable"],
    },
    {
        "kind": "database_saturation",
        "title": "Database saturation cascading into read and payment failures",
        "severity": "P0",
        "affected_services": ["db-primary", "db-replica", "payment-service"],
        "root_cause_service": "db-primary",
        "correct_team": "database",
        "team_map": {
            "db-primary": "database",
            "db-replica": "database",
            "payment-service": "payments-oncall",
        },
        "valid_mitigations": ["scale", "connections", "pool", "kill query"],
    },
    {
        "kind": "payment_rate_limit",
        "title": "Rate-limit cascade across payment processing",
        "severity": "P0",
        "affected_services": ["payment-service", "queue-worker", "api-gateway"],
        "root_cause_service": "payment-service",
        "correct_team": "payments-oncall",
        "team_map": {
            "payment-service": "payments-oncall",
            "queue-worker": "payments-oncall",
            "api-gateway": "platform",
        },
        "valid_mitigations": ["throttle", "backoff", "rate limit", "batch"],
    },
    {
        "kind": "search_index_fault",
        "title": "Search index corruption causing downstream failures",
        "severity": "P1",
        "affected_services": ["search-service", "recommendation-service", "api-gateway"],
        "root_cause_service": "search-service",
        "correct_team": "search",
        "team_map": {
            "search-service": "search",
            "recommendation-service": "search",
            "api-gateway": "platform",
        },
        "valid_mitigations": ["rebuild", "index", "purge", "cache"],
    },
    {
        "kind": "inventory_canary_fault",
        "title": "Canary regression cascading into checkout failures",
        "severity": "P0",
        "affected_services": ["inventory-service", "checkout-service", "api-gateway"],
        "root_cause_service": "inventory-service",
        "correct_team": "backend",
        "team_map": {
            "inventory-service": "backend",
            "checkout-service": "backend",
            "api-gateway": "platform",
        },
        "valid_mitigations": ["rollback", "revert", "restart", "pods"],
    },
]


# ---------------------------------------------------------------------------
# Task 3 — shared log data
# ---------------------------------------------------------------------------

_PAYMENT_CAUSE_LOGS: Dict[str, List[str]] = {
    "db_overload": [
        "WARN  payment-service: query latency p99=3200ms (baseline 120ms)",
        "ERROR payment-service: Deadlock detected on txn_payments table",
        "WARN  db-primary: active connections 490/500 – nearing limit",
        "ERROR payment-service: insert into txn_payments timed out after 5s",
        "INFO  db-primary: autovacuum running on txn_payments (dead tuples: 2.1M)",
        "WARN  payment-service: success_rate=87.3% (SLO: 99.5%)",
    ],
    "rate_limit": [
        "WARN  payment-service: upstream stripe-api returning 429 Too Many Requests",
        "ERROR payment-service: batch charge failed – rate limit exceeded",
        "INFO  payment-service: retrying batch in 2s (backoff attempt 1)",
        "WARN  payment-service: success_rate=91.2% (SLO: 99.5%)",
        "ERROR payment-service: 14 of 50 charges rejected by stripe-api",
        "INFO  payment-service: current request rate 480 req/s (limit: 500)",
    ],
    "memory_leak": [
        "WARN  payment-service: heap usage 3.8GB / 4GB – GC pressure high",
        "ERROR payment-service: OOMKilled by kubelet (exit code 137)",
        "INFO  k8s: pod payment-service-7f8c restarted (restart count: 4)",
        "WARN  payment-service: response latency spike during GC pause (1.8s)",
        "ERROR payment-service: success_rate=92.1% (SLO: 99.5%)",
        "INFO  payment-service: heap growth ~120MB/hr since deploy-4521",
    ],
}

# Shared "noise" logs that appear regardless of root cause
_PAYMENT_NOISE_LOGS: List[str] = [
    "INFO  load-balancer: backend payment-service health check PASS",
    "WARN  monitoring: payment_success_rate dropped below 95% threshold",
    "INFO  payment-service: circuit breaker to downstream-ledger CLOSED",
    "WARN  payment-service: request queue depth 342 (normal: <50)",
]


# ===================================================================
# Main entry point
# ===================================================================

def generate_incident(task_id: str, seed: int) -> dict:
    """Generate a fully-populated incident dict for *task_id*.

    The ``seed`` is applied via a local ``random.Random(seed)`` instance
    **once** here. Inner generators must use that RNG instead of the
    module-global random state.

    Raises:
        ValueError: If *task_id* is not recognised.
    """
    if task_id not in VALID_TASK_IDS:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. "
            f"Valid options: {sorted(VALID_TASK_IDS)}"
        )

    rng = random.Random(seed)

    # Deterministic base epoch for timestamps (varies per seed)
    base_epoch = 1710400000 + (seed % 10000) * 3600

    if task_id == "single_service_outage":
        return _generate_single_service_outage(base_epoch, seed, rng)
    elif task_id == "cascading_failure":
        return _generate_cascading_failure(base_epoch, seed, rng)
    else:
        return _generate_ambiguous_payment_degradation(base_epoch, rng)


# -------------------------------------------------------------------
# Task 1 — single_service_outage
# -------------------------------------------------------------------

def _single_service_logs(
    blueprint: Dict[str, Any],
    base_epoch: int,
    rng: random.Random,
) -> List[str]:
    """Build task 1 logs for the selected single-service scenario."""
    kind = blueprint["kind"]
    service = blueprint["service"]

    if kind == "connection_pool":
        port = 5432 + rng.randint(0, 3)
        waiting = rng.randint(80, 160)
        timeout_ms = rng.randint(2000, 8000)
        retry_count = rng.randint(2, 5)
        retry_logs = [
            f"ERROR {service}: Retry {i}/{retry_count} failed - waiting for connection"
            for i in range(1, retry_count + 1)
        ]
        logs = [
            f"ERROR {service}: ConnectionPool exhausted: db-primary:{port} waiting={waiting}",
            *retry_logs,
            f"WARN  {service}: ConnectionTimeoutError timeout={timeout_ms}ms",
            f"WARN  api-gateway: upstream {service} timeout after 30s",
        ]
    elif kind == "feature_flag":
        deploy_id = f"deploy-{rng.randint(1000, 9999)}"
        flag_name = rng.choice([
            "edge_routing_v2",
            "rate_limit_override",
            "http3_canary",
            "gateway_authz_fastpath",
        ])
        error_rate = rng.randint(35, 92)
        logs = [
            f"WARN  config-service: {deploy_id} enabled feature flag {flag_name} on {service}",
            f"ERROR {service}: FeatureFlagError flag={flag_name} error_rate={error_rate}%",
            f"CRIT  {service}: rollback required for feature flag {flag_name}",
            "WARN  auth-service: downstream token verification requests failing",
        ]
    elif kind == "memory_leak":
        deploy_id = f"deploy-{rng.randint(1000, 9999)}"
        heap_gb = round(rng.uniform(3.6, 4.2), 1)
        restart_count = rng.randint(3, 8)
        logs = [
            f"WARN  {service}: heap usage {heap_gb}GB / 4GB - memory pressure rising",
            f"ERROR {service}: OOMKilled after {deploy_id}",
            f"INFO  k8s: pod {service}-{rng.randint(100, 999)} restarted count={restart_count}",
            f"WARN  deploy-bot: rollback recommended for {service} memory regression",
        ]
    elif kind == "db_saturation":
        connections = rng.randint(940, 1000)
        waiting = rng.randint(60, 180)
        pid = rng.randint(10000, 99999)
        logs = [
            f"ERROR {service}: max_connections reached current={connections}/1000",
            f"WARN  {service}: pool waiters={waiting} - scale connections or kill query",
            f"INFO  {service}: long-running query pid={pid} blocking checkout writes",
            "WARN  api-gateway: upstream requests timing out on primary reads",
        ]
    elif kind == "index_corruption":
        shard = rng.randint(1, 24)
        documents = rng.randint(120000, 980000)
        logs = [
            f"ERROR {service}: corrupt index shard={shard} during search read",
            f"WARN  indexer: purge cache and rebuild index for {service}",
            f"INFO  indexer: last successful segment merge documents={documents}",
            "WARN  api-gateway: downstream search timeout budget exhausted",
        ]
    else:
        deploy_id = f"deploy-{rng.randint(1000, 9999)}"
        canary = rng.choice(["inventory_canary", "stock_sync_v2", "reservation_path"])
        restart_count = rng.randint(3, 7)
        logs = [
            f"WARN  deploy-bot: {deploy_id} enabled canary {canary} for {service}",
            f"ERROR {service}: canary regression detected after {deploy_id}",
            f"CRIT  {service}: rollback or disable canary {canary} immediately",
            f"INFO  k8s: restarted {service}-{rng.randint(100, 999)} count={restart_count}",
        ]

    return _stamp_logs(logs, base_epoch)

def _select_blueprint_index(seed: int, count: int, salt: int) -> int:
    """Return a well-distributed blueprint index for the given seed.
    
    Uses 32-bit Murmur-style mixing to ensure seeds that are close together
    map to different blueprints.
    """
    MASK = 0xFFFFFFFF
    h = (seed ^ salt) & MASK
    h = (((h >> 16) ^ h) * 0x45D9F3B) & MASK
    h = (((h >> 16) ^ h) * 0x45D9F3B) & MASK
    h = ((h >> 16) ^ h) & MASK
    return h % max(count, 1)


def _generate_single_service_outage(
    base_epoch: int,
    seed: int,
    rng: random.Random,
) -> dict:
    """Generate a seed-varying single-service outage."""
    blueprint = _SINGLE_SERVICE_BLUEPRINTS[
        _select_blueprint_index(seed, len(_SINGLE_SERVICE_BLUEPRINTS), 0x15)
    ]
    logs = _single_service_logs(blueprint, base_epoch, rng)

    return {
        "title": blueprint["title"],
        "severity": blueprint["severity"],
        "affected_services": [blueprint["service"]],
        "logs": logs,
        "correct_team": blueprint["correct_team"],
        "team_map": dict(blueprint["team_map"]),
        "valid_mitigations": list(blueprint["valid_mitigations"]),
        "sla_minutes": 30,
    }


# -------------------------------------------------------------------
# Task 2 — cascading_failure
# -------------------------------------------------------------------

def _cascade_logs(
    blueprint: Dict[str, Any],
    base_epoch: int,
    rng: random.Random,
) -> tuple[List[str], Dict[str, Any]]:
    """Build task 2 logs for the selected cascading-failure scenario."""
    kind = blueprint["kind"]
    affected = blueprint["affected_services"]
    root = blueprint["root_cause_service"]

    if kind == "feature_flag_rollout":
        deploy_id = f"deploy-{rng.randint(1000, 9999)}"
        error_rate = rng.randint(45, 95)
        latency_p99 = rng.randint(900, 5000)
        flag_name = rng.choice([
            "enable_new_auth_flow",
            "rate_limit_override",
            "db_pool_size_v2",
            "gateway_shadow_mode",
        ])
        logs = [
            f"WARN  deploy-bot: {deploy_id} rolled out config change flag={flag_name} to {root}",
            f"ERROR {root}: error_rate={error_rate}% after {deploy_id}",
            f"CRIT  {root}: rollback feature flag {flag_name} immediately",
            f"WARN  {affected[1]}: dependency {root} unhealthy - 50% of requests failing",
            f"ERROR {affected[2]}: read traffic spike after {root} degradation",
            f"ERROR {root}: p99_latency={latency_p99}ms - breaching SLO",
        ]
        extra = {"root_cause_deploy": deploy_id, "root_cause_flag": flag_name}
    elif kind == "auth_config_regression":
        deploy_id = f"deploy-{rng.randint(1000, 9999)}"
        config_key = rng.choice([
            "token_verifier_v2",
            "jwt_audience_map",
            "session_cache_bypass",
        ])
        error_rate = rng.randint(40, 90)
        logs = [
            f"WARN  deploy-bot: {deploy_id} pushed config={config_key} to {root}",
            f"ERROR {root}: auth failures spiked to {error_rate}% after {deploy_id}",
            f"CRIT  {root}: rollback config {config_key} or disable the change",
            f"WARN  {affected[1]}: upstream {root} rejecting session validation",
            f"ERROR {affected[2]}: downstream dependency {root} unhealthy",
            "WARN  pagerduty: login traffic degraded across edge and user APIs",
        ]
        extra = {"root_cause_deploy": deploy_id, "root_cause_flag": config_key}
    elif kind == "database_saturation":
        connection_pct = rng.randint(95, 100)
        lag_seconds = rng.randint(8, 20)
        waiting = rng.randint(120, 260)
        logs = [
            f"ERROR {root}: max_connections reached current={connection_pct}%",
            f"CRIT  {root}: scale connections or kill query to recover the pool",
            f"WARN  {affected[1]}: replication lag {lag_seconds}s - read traffic timing out",
            f"ERROR {affected[2]}: dependency {root} pool exhausted waiting={waiting}",
            "WARN  pagerduty: database saturation is cascading to downstream services",
            f"INFO  {root}: slow transaction backlog waiting={waiting}",
        ]
        extra = {}
    elif kind == "payment_rate_limit":
        rate_limited = rng.randint(120, 480)
        queue_depth = rng.randint(300, 1200)
        logs = [
            f"ERROR {root}: upstream 429 Too Many Requests count={rate_limited}",
            f"CRIT  {root}: rate limit exceeded - throttle batch jobs and backoff retries",
            f"WARN  {affected[1]}: retry queue depth={queue_depth} after upstream rejection",
            f"ERROR {affected[2]}: checkout latency elevated while {root} retries backlog",
            "WARN  pagerduty: payment pipeline saturation causing edge impact",
            f"INFO  {root}: current request rate {rng.randint(420, 650)} req/s",
        ]
        extra = {}
    elif kind == "search_index_fault":
        shard = rng.randint(1, 24)
        cache_bytes = rng.randint(128, 640)
        logs = [
            f"ERROR {root}: corrupt index shard={shard} on live query path",
            f"CRIT  {root}: purge cache and rebuild index before retrying traffic",
            f"WARN  {affected[1]}: recommendations degraded because {root} is timing out",
            f"ERROR {affected[2]}: upstream {root} timeout budget exceeded",
            f"INFO  indexer: stale cache footprint={cache_bytes}MB",
            "WARN  pagerduty: search dependency is cascading into serving path",
        ]
        extra = {}
    else:
        deploy_id = f"deploy-{rng.randint(1000, 9999)}"
        restart_count = rng.randint(4, 10)
        canary = rng.choice(["stock_sync_v2", "reservation_path", "inventory_async_write"])
        logs = [
            f"WARN  deploy-bot: {deploy_id} enabled canary {canary} on {root}",
            f"ERROR {root}: canary regression detected after {deploy_id}",
            f"CRIT  {root}: rollback canary {canary} and restart pods",
            f"WARN  {affected[1]}: dependency {root} returning stale inventory",
            f"ERROR {affected[2]}: checkout failures rising with {root} instability",
            f"INFO  k8s: restarted {root}-{rng.randint(100, 999)} count={restart_count}",
        ]
        extra = {"root_cause_deploy": deploy_id, "root_cause_flag": canary}

    return _stamp_logs(logs, base_epoch, interval_minutes=1), extra


def _generate_cascading_failure(
    base_epoch: int,
    seed: int,
    rng: random.Random,
) -> dict:
    """Generate a seed-varying cascading failure across three services."""
    blueprint = _CASCADE_BLUEPRINTS[
        _select_blueprint_index(seed, len(_CASCADE_BLUEPRINTS), 0)
    ]
    logs, extra = _cascade_logs(blueprint, base_epoch, rng)

    incident = {
        "title": blueprint["title"],
        "severity": blueprint["severity"],
        "affected_services": list(blueprint["affected_services"]),
        "root_cause_service": blueprint["root_cause_service"],
        "logs": logs,
        "correct_team": blueprint["correct_team"],
        "team_map": dict(blueprint["team_map"]),
        "valid_mitigations": list(blueprint["valid_mitigations"]),
        "sla_minutes": 15,
    }
    incident.update(extra)
    return incident


# -------------------------------------------------------------------
# Task 3 — ambiguous_payment_degradation
# -------------------------------------------------------------------

def _generate_ambiguous_payment_degradation(
    base_epoch: int,
    rng: random.Random,
) -> dict:
    """Payment degradation with a seed-varying root cause.

    Three possible root causes rotate across seeds so the agent cannot
    memorize the answer. Red herring logs from the other two causes are
    always present to make hypothesis testing genuinely necessary.
    """

    _ROOT_CAUSE_CONFIGS = [
        {
            "root_cause": "db_overload",
            "correct_team": "database",
            "valid_mitigations": ["db", "connection", "pool", "tune"],
            "primary_logs": [
                "WARN  payment-service: query latency p99=3200ms (baseline 120ms)",
                "ERROR payment-service: Deadlock detected on txn_payments table",
                "WARN  db-primary: active connections 490/500 — nearing limit",
                "ERROR payment-service: insert into txn_payments timed out after 5s",
                "WARN  payment-service: success_rate=87.3% (SLO: 99.5%)",
            ],
        },
        {
            "root_cause": "rate_limit",
            "correct_team": "payments-oncall",
            "valid_mitigations": ["backoff", "throttle", "rate", "retry"],
            "primary_logs": [
                "WARN  payment-service: upstream stripe-api returning 429 Too Many Requests",
                "ERROR payment-service: batch charge failed — rate limit exceeded",
                "INFO  payment-service: retrying batch in 2s (backoff attempt 1)",
                "WARN  payment-service: success_rate=91.2% (SLO: 99.5%)",
                "ERROR payment-service: 14 of 50 charges rejected by stripe-api",
            ],
        },
        {
            "root_cause": "memory_leak",
            "correct_team": "payments-oncall",
            "valid_mitigations": ["restart", "rollback", "memory", "heap"],
            "primary_logs": [
                "WARN  payment-service: heap usage 3.8GB / 4GB — GC pressure high",
                "ERROR payment-service: OOMKilled by kubelet (exit code 137)",
                "INFO  k8s: pod payment-service-7f8c restarted (restart count: 4)",
                "WARN  payment-service: response latency spike during GC pause (1.8s)",
                "ERROR payment-service: success_rate=92.1% (SLO: 99.5%)",
            ],
        },
    ]

    # Select root cause using RNG — varies per seed
    selected = rng.choice(_ROOT_CAUSE_CONFIGS)

    # Build red herring logs from the other two causes (2 lines each)
    red_herring_logs = []
    for rc in _ROOT_CAUSE_CONFIGS:
        if rc["root_cause"] != selected["root_cause"]:
            # Take one line from each red herring cause
            red_herring_logs.append(rc["primary_logs"][0])

    noise_logs = [
        "INFO  load-balancer: backend payment-service health check PASS",
        "WARN  monitoring: payment_success_rate dropped below 95% threshold",
        "WARN  payment-service: request queue depth 342 (normal: <50)",
    ]

    all_logs = selected["primary_logs"] + red_herring_logs + noise_logs
    rng.shuffle(all_logs)

    return {
        "title": "Payment latency increased and success rate is degrading",
        "severity": "P1",
        "affected_services": ["payment-service"],
        "logs": _stamp_logs(all_logs[:8], base_epoch),
        "root_cause": selected["root_cause"],
        "correct_team": selected["correct_team"],
        "team": selected["correct_team"],
        "team_map": {
            "payment-service": "payments-oncall",
            "db-primary": "database",
            "stripe-api": "payments-oncall",
        },
        "valid_mitigations": selected["valid_mitigations"],
        "hypotheses": ["db_overload", "rate_limit", "memory_leak"],
        "sla_minutes": 45,
        "initial_metrics": [
            {
                "service": "payment-service",
                "error_rate_pct": 40.0,
                "latency_p99_ms": 2100,
                "throughput_rps": 320,
                "status": "degraded",
            }
        ],
    }