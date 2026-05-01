"""
Inference script for the Incident Response Environment.

Connects to a running environment server, uses an OpenAI-compatible LLM
endpoint to generate actions, and runs **all 3 tasks sequentially**:
    1. single_service_outage
    2. cascading_failure
    3. ambiguous_payment_degradation

Environment variables (required):
    API_BASE_URL  – OpenAI-compatible base URL
    MODEL_NAME    – Model identifier
    HF_TOKEN      – Bearer token / API key  (used as api_key)
    ENV_URL       – Environment server URL   (default: http://localhost:7860)

Log format:
    [START] task=<id> env=<url> model=<name>
    [STEP]  step=<n> action=<repr> reward=<f> done=<bool> error=<str|None>
    [END]   success=<bool> steps=<n> score=<f> rewards=<list>
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple
# from dotenv import load_dotenv
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # env vars must be set directly in the environment
# Ensure project root is importable so we can use server.tasks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.tasks import TASK_CONFIGS


_local_env = None

def _get_local_env():
    global _local_env
    if _local_env is None:
        from server.env import IncidentResponseEnv
        _local_env = IncidentResponseEnv()
    return _local_env


def _obs_to_dict(obs):
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return obs.__dict__

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "openai/gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
ENV_URL: str = os.environ.get("ENV_URL", "")
USE_TRAINED_MODEL: bool = os.environ.get("USE_TRAINED_MODEL", "1") == "1"

# Direct in-process env calls — no HTTP server needed.
# True by default: only falls back to HTTP when ENV_URL is explicitly set.
USE_LOCAL_ENV: bool = (
    os.environ.get("USE_LOCAL_ENV", "1") == "1"
    or not ENV_URL
)

# Trained LoRA model id on Hugging Face (required when USE_TRAINED_MODEL=1)
MODEL_ID: str | None = os.environ.get("MODEL_ID")

# Short env name for [START] log line (PS spec requires an identifier, not a URL)
ENV_NAME: str = "incident_response_env"

# Fixed seed for reproducibility
FIXED_SEED: int = 42

# Task sequence — matches TASK_CONFIGS keys in server/tasks.py
TASK_IDS: List[str] = [
    "single_service_outage",
    "cascading_failure",
    "ambiguous_payment_degradation",
]

MAX_TOTAL_REWARDS = {
    "single_service_outage": 1.00,
    "cascading_failure": 1.00,
    "ambiguous_payment_degradation": 1.00,
}

MAX_STEPS = {
    "single_service_outage": 12,   # was 10
    "cascading_failure": 18,
    "ambiguous_payment_degradation": 25,
}

# ---------------------------------------------------------------------------
# Load trained model (local, via Unsloth)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None

# Default 4-bit base model — the LoRA adapter sits on top of this.
_BASE_MODEL = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"


def get_model():
    """Load the base model + LoRA adapter in HF-safe fashion.

    Strategy:
      1. Load the 4-bit quantised base model with Unsloth
      2. Load the LoRA adapter separately (no merge — avoids OOM on HF GPU)
      3. Switch to inference mode via FastLanguageModel.for_inference

    MODEL_ID must point to a HF repo containing LoRA adapter weights
    (adapter_config.json + adapter_model.safetensors).
    """
    global _model, _tokenizer

    if _model is None:
        if not MODEL_ID:
            raise ValueError(
                "MODEL_ID not set. Set MODEL_ID to your pushed LoRA repo "
                "(e.g. rehannaveid/incidentiq-lora) once training finishes.\n"
                "Run with USE_TRAINED_MODEL=0 to use the OpenAI-compatible API instead."
            )

        try:
            from unsloth import FastLanguageModel
        except Exception as e:
            raise RuntimeError(
                "Failed to import `unsloth`. Install it with: pip install unsloth"
            ) from e

        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "Failed to import `peft`. Install it with: pip install peft"
            ) from e

        print(f"[INFER] Loading base model {_BASE_MODEL} (4-bit) ...", flush=True)
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=_BASE_MODEL,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,   # auto: bfloat16 on A100, float16 elsewhere
        )

        print(f"[INFER] Loading LoRA adapter from {MODEL_ID} ...", flush=True)
        _model = PeftModel.from_pretrained(_model, MODEL_ID)
        # ❌ DO NOT merge — merge_and_unload() causes OOM on HF GPU instances
        # _model = _model.merge_and_unload()

        FastLanguageModel.for_inference(_model)
        print("[INFER] Model ready.", flush=True)

    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Phase 2 scoring weights
# ---------------------------------------------------------------------------

R1_WEIGHT = 0.60   # Task accuracy — existing graders (grade_task1/2/3)
R2_WEIGHT = 0.40   # Belief calibration — cross-entropy −Σ p(s) log p̂(s)

SYSTEM_PROMPT = """\
You are an autonomous incident response agent operating in a distributed system.

Your goal is to identify the TRUE root cause of an incident and resolve it efficiently.

You are trained using reinforcement learning. Your performance is evaluated based on:
1. Correct root cause identification and resolution  (R1 — task accuracy,   weight 0.60)
2. Accurate belief updates over time                (R2 — belief calibration, weight 0.40)

You MUST reason under uncertainty and update your belief at every step.

You DO NOT get the correct answer from the environment.
You MUST infer it from logs, metrics, and feedback.

STRICT RULES:
  - NEVER investigate a service not listed in AFFECTED SERVICES
  - NEVER investigate the same service twice
  - NEVER invent a team name; use only names from TEAM ROSTER
  - NEVER skip steps: investigate → assign team → mitigate → resolve
  - ONE action per turn, nothing else
  - If mitigation fails, revise your belief and try a different fix
  - If LIVE METRICS are healthy after mitigation, your next action MUST be resolve

Always follow the rules strictly.
"""


# ---------------------------------------------------------------------------
# Mitigate detection helper  (used by build_prompt)
# ---------------------------------------------------------------------------

MITIGATE_ACTION_PREFIXES = (
    "mitigate:",
    "restart",
    "rollback",
    "revert",
    "throttle",
    "backoff",
    "vacuum",
    "disable",
    "block",
    "isolate",
    "fix",
)


def _is_mitigate(action: str) -> bool:
    """Check if an action string is an explicit mitigation action."""
    a = action.lower().strip()
    return a.startswith(MITIGATE_ACTION_PREFIXES)


def _service_investigated(action_history: List[str], service: str) -> bool:
    """Return True when the service was investigated in a prior action."""
    service_lower = service.lower()
    return any(
        "investigate" in action.lower() and service_lower in action.lower()
        for action in action_history
    )


def _investigated_services(action_history: List[str], affected: List[str]) -> List[str]:
    """Return affected services that have already been investigated."""
    return [svc for svc in affected if _service_investigated(action_history, svc)]


# ---------------------------------------------------------------------------
# Logging helpers  (exact format — do not deviate)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    done_str  = "true" if done else "false"
    error_str = error if error is not None else "null"
    action_clean = (
        action.strip().strip('"').strip("'")
              .replace("\n", " ").replace("\r", "")
              .replace("=", "-")
              .strip()
    )
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Env-var validation
# ---------------------------------------------------------------------------

def _require_env(name: str, value: str) -> str:
    if not value:
        print(f"[ERROR] Environment variable {name} is required.", file=sys.stderr)
        sys.exit(1)
    return value


# ---------------------------------------------------------------------------
# LLM client builder
# ---------------------------------------------------------------------------

def _build_client():
    """Build OpenAI client with provider-specific headers."""
    try:
        from openai import OpenAI  # local import so training can import inference.py without openai installed
    except Exception as e:
        raise RuntimeError(
            "OpenAI client not available. Install `openai` or set USE_TRAINED_MODEL=1."
        ) from e

    extra_headers = {}
    if "openrouter" in API_BASE_URL.lower():
        extra_headers = {
            "HTTP-Referer": "https://github.com/incident-response-env",
            "X-Title": "Incident Response Environment",
        }
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        default_headers=extra_headers if extra_headers else None,
    )


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def ask_llm_openai(
    client,
    observation: Dict[str, Any],
    history: List[Dict],
    action_history: List[str],
) -> Tuple[str, str, str]:
    """Build a prompt, include conversation history, call the LLM.

    Returns (action_str, reasoning_str, raw_response_text).
    reasoning_str packs Thought + Belief so env._parse_belief_from_reasoning()
    can extract the belief for the R2 grader.

    Belief is NOT re-injected into the prompt — the agent must reconstruct
    it from the current observation each step (belief-as-output, not input).
    The stability reward in environment.py incentivises smooth updates.
    """
    obs_prompt = build_prompt(observation, action_history)
    default_service = next(
        iter(observation.get("affected_services", []) or []),
        "auth-service",
    )
    fallback_action = f"investigate {default_service}"

    try:
        # Cap history to last 6 exchanges (12 messages) to avoid token limits
        recent_history = history[-12:] if len(history) > 12 else history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent_history
        messages.append({"role": "user", "content": format_prompt(observation, recent_history)})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.8,   # Phase 3: exploration required for RL / GRPO
            max_tokens=512,
        )
        response_text = (response.choices[0].message.content or "").strip()
        if not response_text:
            return fallback_action, "", ""

        # Strip markdown code fences if present
        cleaned = response_text
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```[\w]*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
            cleaned = cleaned.strip()

        # Parse JSON response
        try:
            parsed = json.loads(cleaned)
            thought_str = parsed.get("thought", "").strip()
            belief_dict = parsed.get("belief", {})
            candidates = (
                observation.get("fan_in_candidates")
                or observation.get("hypotheses")
                or observation.get("affected_services", [])
            )
            if isinstance(belief_dict, dict) and candidates:
                belief_dict = {
                    key: (
                        max(0.0, float(belief_dict.get(key, 0.0)))
                        if isinstance(belief_dict.get(key, 0.0), (int, float))
                        else 0.0
                    )
                    for key in candidates
                }
                total = sum(belief_dict.values())
                if total > 0.0:
                    belief_dict = {
                        key: value / total
                        for key, value in belief_dict.items()
                    }
                else:
                    belief_dict = {key: 1.0 / len(candidates) for key in candidates}
            action_str = parsed.get("action", "").strip()
            # Pack both into reasoning so env._parse_belief_from_reasoning() can find it
            reasoning_str = f"Thought: {thought_str}\nBelief: {json.dumps(belief_dict)}"
        except Exception:
            # Fallback: treat first line as action, no reasoning
            action_str = cleaned.splitlines()[0].strip().strip('"').strip("'")
            reasoning_str = ""

        if not action_str:
            action_str = fallback_action

        return action_str, reasoning_str, response_text
    except Exception:
        return fallback_action, "", ""


def ask_llm_local(
    observation: Dict[str, Any],
    action_history: List[str],
    last_belief: Dict[str, float],
    last_action: str,
) -> Tuple[str, str, Dict[str, float]]:
    """Generate action using trained local model."""
    # Lazy import: baseline mode should not require torch/unsloth installed.
    from utils import format_stateful_prompt, generate, parse_output

    model, tokenizer = get_model()

    prompt = format_stateful_prompt(observation, last_belief, last_action)

    try:
        raw = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )
    except Exception as e:
        print(f"[INFER] generate failed: {e}", flush=True)
        affected = observation.get("affected_services", ["auth-service"])
        return f"investigate {affected[0]}", "", {}

    affected = observation.get("affected_services", [])
    belief_candidates = (
        observation.get("fan_in_candidates")
        or observation.get("hypotheses")
        or affected
    )
    action, reasoning, belief, _ = parse_output(raw, affected, belief_candidates)
    return action, reasoning, belief


# ---------------------------------------------------------------------------
# Phase 3 — Training prompt
# ---------------------------------------------------------------------------

def format_prompt(
    observation: Dict[str, Any],
    history: List[Dict],
    action_history: List[str] | None = None,
) -> str:
    """Build the Phase 3 RL training prompt for each step.

    Composes a five-section template around the rich observation text
    produced by ``build_prompt``.  This is the complete user-turn message
    the LLM sees at every step.

    Design goals:
      - Enforces strict JSON output structure  (parsing stability)
      - Forces explicit belief updates         (R2 learning signal)
      - Directly mentions R1/R2 reward weights (RL alignment)
      - Shows previous steps inline            (sequential reasoning)

    Args:
        observation:    Raw observation dict from the environment /step response.
        history:        Conversation history list (role/content dicts).
        action_history: Optional list of raw action strings taken so far.
                        When provided (training), build_prompt uses them for
                        NEXT hints, hypothesis tracking, and resolve_ready.
                        When None (inference), defaults to empty list.

    Returns:
        Formatted user prompt string.
    """
    # Thread action_history into build_prompt so resolve_ready, hypothesis
    # checklist, and NEXT directives fire correctly during training rollouts.
    obs_section = build_prompt(observation, action_history or [])

    affected     = observation.get("affected_services", [])
    affected_str = ", ".join(affected) if affected else "(see logs)"

    # Render last 4 exchanges as compact text.
    # Assistant turns: extract only the action field so the model sees clean
    # history without malformed mid-truncated JSON.
    # User (env) turns: first 200 chars is enough for feedback context.
    history_lines: List[str] = []
    for msg in history:
        role    = msg.get("role", "")
        content = msg.get("content") or ""
        if role == "assistant":
            try:
                parsed = json.loads(content)
                history_lines.append(f"[YOU] action: {parsed.get('action', '?')}")
            except Exception:
                history_lines.append(f"[YOU] {content[:120]}")
        elif role == "user":
            history_lines.append(f"[ENV] {content[:200]}")
    history_text = (
        "\n".join(history_lines[-8:])
        if history_lines
        else "(first step — no history yet)"
    )

    # Belief keys must match what R2 grades against:
    #   Task 1/2: fan_in_candidates    Task 3: hypotheses
    belief_candidates = (
        observation.get("fan_in_candidates")
        or observation.get("hypotheses")
        or affected
    )
    candidate_list = ", ".join(belief_candidates) if belief_candidates else "(none)"
    belief_keys = ",  ".join(f'"{s}": <probability>' for s in belief_candidates)

    return f"""\
You are solving a live production incident.

=====================
CURRENT OBSERVATION
=====================
{obs_section}

=====================
YOUR PREVIOUS STEPS
=====================
{history_text}

=====================
INSTRUCTIONS
=====================

You must perform ONE step of reasoning and action.

STEP 1 — ANALYZE
- Read logs and metrics carefully
- Identify signals vs noise
- Do NOT assume the first error is the root cause

STEP 2 — UPDATE BELIEF
- Maintain a probability distribution over the belief keys shown in the JSON template
- Your belief must ONLY contain these keys: {candidate_list}
- You MUST assign probability to ALL candidates
- No extra keys are allowed
- Belief values must sum to approximately 1.0
- Increase probability for a service when evidence implicates it
- Decrease probability when investigation shows the service is healthy
- If you investigated a service and found no issue, its probability MUST decrease significantly

STEP 3 — DECIDE ACTION
You may ONLY choose one of:
  - investigate <service>
  - assign to <team>
  - mitigate: <fix>
  - resolve

Rules:
- ONLY investigate services in AFFECTED SERVICES
- NEVER repeat investigation of the same service
- ONLY mitigate AFTER investigation
- ONLY resolve when LIVE METRICS confirm the system is healthy
- If mitigation fails → revise belief, try a different fix

=====================
OUTPUT FORMAT (STRICT)
=====================

Return ONLY valid JSON. No prose before or after:

{{
  "thought": "step-by-step reasoning referencing specific log lines and previous actions",
  "belief": {{{belief_keys}}},
  "action": "one valid action"
}}

=====================
CRITICAL CONSTRAINTS
=====================

- belief MUST be a valid probability distribution (values sum to ~1.0)
- belief MUST include every listed belief key and no other keys
- belief MUST change based on new evidence each step
- thought MUST cite specific log lines as evidence
- do NOT output text outside the JSON block
- do NOT repeat the same thought two steps in a row
- do NOT guess randomly

If your belief is incorrect, your R2 reward will be reduced.
If your action is incorrect, your R1 reward will be reduced.
Your goal is to maximise total reward by:
  • correctly identifying and resolving the root cause   (R1 weight 0.60)
  • updating belief accurately at every step             (R2 weight 0.40)

Now produce your next step.
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    observation: Dict[str, Any],
    action_history: List[str] | None = None,
) -> str:
    """Build a user prompt from the environment observation dict."""
    action_history = action_history or []
    obs = observation
    affected = obs.get("affected_services", [])
    task_id = obs.get("task_id", "")
    feedback = obs.get("feedback", "")
    feedback_lower = feedback.lower()
    mitigation_failed = (
        "no matching keywords found" in feedback_lower
        or "mitigation failed" in feedback_lower
    )
    severity = obs.get("severity", "")
    sla = obs.get("sla_remaining", "?")

    metrics = obs.get("metrics", [])
    mitigation_applied = "fix applied" in feedback_lower
    metrics_recovered = (
        "live metrics are now healthy" in feedback_lower
        or "incident resolved successfully" in feedback_lower
        or (
            bool(metrics)
            and all(
                float(m.get("error_rate_pct", 100.0)) <= 2.0
                and str(m.get("status", "degraded")).lower() == "ok"
                for m in metrics
                if isinstance(m, dict)
            )
        )
    )
    resolve_ready = metrics_recovered and (mitigation_applied or any(_is_mitigate(a) for a in action_history))
    resolved = bool(obs.get("resolved", False))

    investigated = _investigated_services(action_history, affected)
    remaining = [svc for svc in affected if svc not in investigated]
    all_investigated = not remaining if affected else False

    hypothesis_checks = [
        (
            "investigate payment-service db connection",
            ("db", "database", "connection", "query", "deadlock"),
        ),
        (
            "investigate payment-service rate limit",
            ("rate", "stripe", "429", "throttle", "upstream"),
        ),
        (
            "investigate payment-service memory heap",
            ("memory", "heap", "oom", "leak", "gc"),
        ),
    ]
    hypothesis_statuses: List[tuple[str, bool]] = []
    if task_id == "ambiguous_payment_degradation":
        for action_text, keywords in hypothesis_checks:
            done = any(
                "investigate" in a.lower()
                and "payment-service" in a.lower()
                and any(kw in a.lower() for kw in keywords)
                for a in action_history
            )
            hypothesis_statuses.append((action_text, done))
    all_hypotheses_investigated = (
        all(done for _, done in hypothesis_statuses)
        if hypothesis_statuses else True
    )
    ready_to_mitigate = all_investigated and all_hypotheses_investigated

    lines = [
        f"TASK: {task_id}",
        f"SEVERITY: {severity} | SLA REMAINING: {sla} minutes",
        "",
    ]

    if feedback:
        lines.extend([
            "=== RESULT OF YOUR LAST ACTION ===",
            feedback,
            "==================================",
            "",
        ])

    lines.extend([
        "INCIDENT:",
        obs.get("incident_description", "(no description)"),
        "",
        "AFFECTED SERVICES (only investigate these, nothing else):",
    ])
    for svc in affected:
        lines.append(f"  - {svc}")

    if investigated:
        lines.extend([
            "",
            f"ALREADY INVESTIGATED: {', '.join(investigated)}",
        ])
        if remaining:
            lines.append(f"NOT YET INVESTIGATED: {', '.join(remaining)}")
        else:
            lines.append("ALL SERVICES INVESTIGATED - do not investigate further.")

    if task_id == "ambiguous_payment_degradation":
        lines.extend([
            "",
            "HYPOTHESIS CHECKLIST — investigate each separately using the EXACT text below:",
        ])
        for action_text, done in hypothesis_statuses:
            status = "DONE" if done else "PENDING"
            lines.append(f"  [{status}] {action_text}")
        if not all_hypotheses_investigated:
            lines.append("  !! DO NOT say 'investigate payment-service' — use the EXACT hypothesis text above !!")

    lines.extend([
        "",
        "LOGS (use these to infer the most likely root cause):",
    ])
    for log_line in obs.get("logs", []):
        lines.append(f"  {log_line}")

    # Show live metrics if available
    if metrics:
        lines.extend([
            "",
            "LIVE METRICS:",
        ])
        for m in metrics:
            if isinstance(m, dict):
                lines.append(
                    f"  {m.get('service', '?')}: "
                    f"error_rate={m.get('error_rate_pct', '?')}% "
                    f"latency_p99={m.get('latency_p99_ms', '?')}ms "
                    f"throughput={m.get('throughput_rps', '?')}rps "
                    f"[{m.get('status', '?')}]"
                )

    # Show available teams
    team_roster = obs.get("team_roster", {})
    if team_roster:
        available_teams = [
            team for team, status in team_roster.items()
            if str(status).lower() == "available"
        ]
        lines.extend([
            "",
            "TEAM ROSTER (use EXACTLY these team names in assign actions):",
        ])
        for team, status in team_roster.items():
            marker = "<- USE THIS NAME" if str(status).lower() == "available" else ""
            lines.append(f"  {team}: {status} {marker}".rstrip())
        if available_teams:
            lines.append(f"AVAILABLE TEAM NAMES: {', '.join(available_teams)}")

    if action_history:
        lines.extend([
            "",
            "ACTIONS TAKEN SO FAR (do not repeat these):",
        ])
        for i, a in enumerate(action_history, 1):
            lines.append(f"  {i}. {a}")

    lines.append("")

    # Tell the model exactly what to do next
    if resolve_ready and not resolved:
        lines.append("RESULT AND LIVE METRICS CONFIRM RECOVERY - your only valid next action is: resolve")
    elif mitigation_applied and not metrics_recovered:
        last_mitigation = next(
            (a for a in reversed(action_history) if _is_mitigate(a)),
            "mitigate: <repeat the successful fix>",
        )
        lines.append(
            f"MITIGATION APPLIED BUT METRICS STILL DEGRADED. "
            f"Your ONLY valid next action is: {last_mitigation}"
        )
    elif mitigation_failed:
        lines.append("YOUR LAST MITIGATION FAILED - do not resolve. Reassess the logs, LIVE METRICS, and your hypothesis before trying another mitigation.")
    elif "cannot resolve" in feedback_lower:
        if "still degraded" in feedback_lower:
            last_mitigation = next(
                (a for a in reversed(action_history) if _is_mitigate(a)),
                "mitigate: <repeat the successful fix>",
            )
            lines.append(
                f"RESOLVE FAILED - the affected services are still degraded. Re-apply the same mitigation only if the evidence still supports it and watch LIVE METRICS: {last_mitigation}"
            )
        else:
            lines.append("RESOLVE FAILED - no valid mitigation has been confirmed yet. Apply a mitigation that best fits the evidence.")
    elif "team name not recognized" in feedback_lower:
        lines.append("YOUR LAST TEAM ASSIGNMENT WAS INVALID - retry using EXACTLY one name from TEAM ROSTER.")
    elif task_id == "ambiguous_payment_degradation" and not all_hypotheses_investigated:
        pending_hypothesis = next(
            (action_text for action_text, done in hypothesis_statuses if not done),
            "investigate payment-service db connection",
        )
        lines.append(
            f"\u26a0\ufe0f MANDATORY NEXT ACTION: {pending_hypothesis}\r\n"
            f"  (copy-paste this EXACTLY \u2014 do NOT say 'investigate payment-service' without the suffix)"
        )
    elif ready_to_mitigate:
        lines.append("ALL REQUIRED INVESTIGATIONS ARE COMPLETE - choose the owner from TEAM ROSTER and the mitigation that best fits the evidence.")
    else:
        if remaining:
            lines.append(f"NEXT: investigate {remaining[0]}")
        else:
            lines.append("ALL SERVICES ARE ALREADY INVESTIGATED - do not investigate further. Assign a team or apply a mitigation.")

    lines.extend([
        "",
        "What is your next single action?",
        "(investigate <service> | assign to <team> | mitigate: <fix> | escalate | resolve)",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session management helpers
# ---------------------------------------------------------------------------

def _session_headers(session_id: str) -> Dict[str, str]:
    """Return headers dict with X-Session-Id."""
    return {"X-Session-Id": session_id}


def _fetch_episode_r2(session_headers: Dict[str, str]) -> float:
    """Fetch R2 cross-entropy calibration score from /state after the episode.

    Returns the server-computed r2_score in [0.0, 1.0].
    Returns 0.0 on any network or parse error (safe default).
    """
    try:
        resp = requests.get(
            f"{ENV_URL}/state", headers=session_headers, timeout=10
        )
        resp.raise_for_status()
        r2 = resp.json().get("info", {}).get("r2_score", 0.0)
        return float(r2 or 0.0)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Environment interaction helpers (HTTP)
# ---------------------------------------------------------------------------

def get_seed_meta(
    task_id: str,
    observation: dict = None,
    session_id: str = "",
) -> dict:
    headers = {"X-Session-Id": session_id} if session_id else {}
    seed_meta = {}

    try:
        resp = requests.get(
            f"{ENV_URL}/incident-meta", headers=headers, timeout=10
        )
        resp.raise_for_status()
        seed_meta = resp.json()
    except Exception:
        pass

    # Secondary fallback — /state carries info{} with same keys
    if not seed_meta.get("valid_mitigations"):
        try:
            resp = requests.get(
                f"{ENV_URL}/state", headers=headers, timeout=10
            )
            resp.raise_for_status()
            state = resp.json()
            seed_meta = {**seed_meta, **state.get("info", {})}
        except Exception:
            pass

    # Final fallback — observation carries some fields directly
    if not seed_meta.get("valid_mitigations") and observation:
        seed_meta.setdefault("correct_team",      observation.get("correct_team", ""))
        seed_meta.setdefault("valid_mitigations", observation.get("valid_mitigations", []))
        seed_meta.setdefault("affected_services", observation.get("affected_services", []))

    return seed_meta


def env_step(action: str, reasoning: str = "") -> Dict[str, Any]:
    """POST /step with an IncidentAction payload."""
    payload = {
        "action": {
            "action": action,
            "reasoning": reasoning,
        }
    }
    resp = requests.post(f"{ENV_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    observation = data.get("observation", {})
    if (
        isinstance(observation, dict)
        and "observation" in observation
        and isinstance(observation.get("observation"), dict)
    ):
        # Backward compatibility for servers that incorrectly return StepResult
        # instead of an Observation from env.step().
        flattened = dict(data)
        flattened["observation"] = observation["observation"]
        if "info" in observation and "info" not in flattened:
            flattened["info"] = observation["info"]
        return flattened
    return data


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, client=None, seed: int = FIXED_SEED) -> Dict[str, Any]:
    max_steps        = MAX_STEPS.get(task_id, 10)
    active_model_name = (MODEL_ID or "MODEL_ID_UNSET") if USE_TRAINED_MODEL else MODEL_NAME
    log_start(task=task_id, model=active_model_name)

    step_num        = 0
    rewards: List[float]    = []
    action_history: List[str]  = []
    history:   List[Dict]   = []
    reasoning_texts: List[str] = []
    score   = 0.0
    success = False
    seed_meta: dict = {}

    try:
        if USE_LOCAL_ENV:
            # ----------------------------------------------------------------
            # Direct-env path — no HTTP, no server needed
            # ----------------------------------------------------------------
            from models import IncidentAction as _IncidentAction

            env = _get_local_env()

            try:
                obs_obj = env.reset(task_id=task_id, seed=seed)
            except Exception as exc:
                log_step(step=0, action="reset", reward=0.0, done=True, error=str(exc))
                return {"task_id": task_id, "success": False, "steps": 0,
                        "score": 0.0, "rewards": []}

            observation = _obs_to_dict(obs_obj)
            # Seed meta comes straight from env._incident_data — zero latency
            seed_meta = {k: env._incident_data.get(k, "")
                         for k in ("root_cause", "team", "correct_team",
                                   "valid_mitigations", "root_cause_service",
                                   "affected_services")}
            done = obs_obj.done
            last_belief_dict: Dict[str, float] = {}

            while not done and step_num < max_steps:
                step_num += 1
                error: str | None = None

                if USE_TRAINED_MODEL:
                    chosen_action, reasoning_text, last_belief_dict = ask_llm_local(
                        observation, action_history, history, last_belief_dict, ""
                    )
                else:
                    chosen_action, reasoning_text, raw_response = ask_llm_openai(
                        client, observation, history, action_history,
                    )
                    try:
                        parsed_resp = json.loads(raw_response) if raw_response else {}
                        last_belief_dict = parsed_resp.get("belief", last_belief_dict)
                    except Exception:
                        pass

                if not last_belief_dict:
                    print(f"[WARN] step={step_num} no belief — R2=0 this step", flush=True)

                action_history.append(chosen_action)
                try:
                    history.append({"role": "assistant",
                                    "content": json.dumps({"action": chosen_action,
                                                           "thought": "", "belief": last_belief_dict})})
                except Exception:
                    history.append({"role": "assistant", "content": chosen_action})

                try:
                    obs_obj = env.step(_IncidentAction(action=chosen_action,
                                                       reasoning=reasoning_text))
                    observation = _obs_to_dict(obs_obj)
                    done   = obs_obj.done
                    reward = float(obs_obj.reward or 0.0)
                    feedback = observation.get("feedback", "")
                except Exception as exc:
                    error  = str(exc)
                    reward = 0.0
                    done   = True
                    feedback = ""

                result_summary = f"RESULT OF YOUR LAST ACTION: {feedback} Reward={reward:.2f}."
                if "Fix applied" in feedback:
                    if "healthy" in feedback or "you may resolve" in feedback.lower():
                        result_summary += " LIVE METRICS healthy — resolve next."
                    else:
                        result_summary += " Still degraded — re-apply mitigation."
                elif "Cannot resolve" in feedback:
                    result_summary += " Apply mitigation first."
                elif "Team name not recognized" in feedback:
                    result_summary += " Retry with exact team name from roster."
                history.append({"role": "user", "content": result_summary})
                if len(history) > 12:
                    history = history[-12:]

                rewards.append(reward)
                log_step(step=step_num, action=chosen_action,
                         reward=reward, done=done, error=error)
                if reasoning_text:
                    reasoning_texts.append(reasoning_text)
                if done:
                    break

            # Grading — read R2 directly from env.state() (no HTTP)
            task_config = TASK_CONFIGS.get(task_id, {})
            grader = task_config.get("grader")
            if grader:
                r1_score = grader(action_history, seed_meta,
                                  reasoning_texts=reasoning_texts)
                r2_score = 0.0
                if last_belief_dict:
                    try:
                        st = env.state()
                        r2_score = float(
                            (st.model_dump() if hasattr(st, "model_dump") else st.dict())
                            .get("info", {}).get("r2_score", 0.0) or 0.0
                        )
                    except Exception:
                        pass
                score = max(0.0, min(1.0, R1_WEIGHT * r1_score + R2_WEIGHT * r2_score))
                print(
                    f"[GRADE] task={task_id} R1={r1_score:.4f} R2={r2_score:.4f} "
                    f"score={score:.4f} beliefs={len(reasoning_texts)}/{step_num}",
                    flush=True,
                )
            else:
                score = float(observation.get("score", 0.0) or 0.0)

        else:
            # ----------------------------------------------------------------
            # HTTP path — env runs as a separate server (HF Space or local)
            # ----------------------------------------------------------------
            import uuid
            session_id = str(uuid.uuid4())
            session_headers = {"X-Session-Id": session_id}

            try:
                resp = requests.post(
                    f"{ENV_URL}/reset",
                    json={"task_id": task_id, "seed": seed},
                    headers=session_headers, timeout=30,
                )
                resp.raise_for_status()
                reset_resp = resp.json()
            except Exception as exc:
                log_step(step=0, action="reset", reward=0.0, done=True, error=str(exc))
                return {"task_id": task_id, "success": False, "steps": 0,
                        "score": 0.0, "rewards": []}

            observation = reset_resp.get("observation", {})
            seed_meta   = get_seed_meta(task_id, observation, session_id=session_id)
            done        = reset_resp.get("done", False)
            last_belief_dict: Dict[str, float] = {}

            while not done and step_num < max_steps:
                step_num += 1
                error: str | None = None

                if USE_TRAINED_MODEL:
                    chosen_action, reasoning_text, last_belief_dict = ask_llm_local(
                        observation, action_history, history, last_belief_dict, ""
                    )
                else:
                    chosen_action, reasoning_text, raw_response = ask_llm_openai(
                        client, observation, history, action_history,
                    )
                    try:
                        parsed_resp = json.loads(raw_response) if raw_response else {}
                        last_belief_dict = parsed_resp.get("belief", last_belief_dict)
                    except Exception:
                        pass

                if not last_belief_dict:
                    print(f"[WARN] step={step_num} no belief — R2=0", flush=True)

                action_history.append(chosen_action)
                history.append({"role": "assistant", "content": chosen_action})

                try:
                    resp = requests.post(
                        f"{ENV_URL}/step",
                        json={"action": {"action": chosen_action, "reasoning": reasoning_text}},
                        headers=session_headers, timeout=30,
                    )
                    resp.raise_for_status()
                    step_resp   = resp.json()
                    observation = step_resp.get("observation", {})
                    done   = step_resp.get("done", False)
                    reward = float(step_resp.get("reward", 0.0) or 0.0)
                    feedback = observation.get("feedback", "")
                    result_summary = f"RESULT OF YOUR LAST ACTION: {feedback} Reward={reward:.2f}."
                    if "Fix applied" in feedback:
                        result_summary += (" LIVE METRICS healthy — resolve next."
                                           if "healthy" in feedback
                                           else " Still degraded — re-apply mitigation.")
                    elif "Cannot resolve" in feedback:
                        result_summary += " Apply mitigation first."
                    elif "Team name not recognized" in feedback:
                        result_summary += " Retry with exact team name from roster."
                    history.append({"role": "user", "content": result_summary})
                except Exception as exc:
                    error  = str(exc)
                    reward = 0.0
                    done   = True

                rewards.append(reward)
                log_step(step=step_num, action=chosen_action,
                         reward=reward, done=done, error=error)
                if reasoning_text:
                    reasoning_texts.append(reasoning_text)
                if done:
                    break

            task_config = TASK_CONFIGS.get(task_id, {})
            grader = task_config.get("grader")
            if grader:
                r1_score = grader(action_history, seed_meta,
                                  reasoning_texts=reasoning_texts)
                r2_score = _fetch_episode_r2(session_headers)
                if not last_belief_dict:
                    r2_score = 0.0
                score = max(0.0, min(1.0, R1_WEIGHT * r1_score + R2_WEIGHT * r2_score))
                print(
                    f"[GRADE] task={task_id} R1={r1_score:.4f} R2={r2_score:.4f} "
                    f"score={score:.4f} beliefs={len(reasoning_texts)}/{step_num}",
                    flush=True,
                )
            else:
                score = float(observation.get("score", 0.0) or 0.0)

            try:
                requests.delete(f"{ENV_URL}/session", headers=session_headers, timeout=5)
            except Exception:
                pass

        score   = max(0.0, min(score, 1.0))
        success = score >= 0.5

    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)

    finally:
        log_end(success=success, steps=step_num, score=score, rewards=rewards)

    return {"task_id": task_id, "success": success, "steps": step_num,
            "score": score, "rewards": rewards}

# ---------------------------------------------------------------------------
# Main — run all 3 tasks sequentially
# ---------------------------------------------------------------------------

def main() -> None:
    client = None
    if USE_TRAINED_MODEL:
        # No API keys needed — local model
        pass
    else:
        _require_env("API_BASE_URL", API_BASE_URL)
        _require_env("MODEL_NAME", MODEL_NAME)
        _require_env("HF_TOKEN", HF_TOKEN)
        client = _build_client()

    results: List[Dict[str, Any]] = []
    task_ids_override = os.environ.get("TASK_IDS_OVERRIDE", "").strip()
    if task_ids_override:
        task_ids = [task_id.strip() for task_id in task_ids_override.split(",") if task_id.strip()]
    else:
        task_ids = TASK_IDS

    for task_id in task_ids:
        result = run_task(task_id, client, seed=FIXED_SEED)
        results.append(result)

    # ---- Overall summary with visual score bars ----
    total_score = sum(r["score"] for r in results) / len(results)
    total_steps = sum(r["steps"] for r in results)
    all_success = all(r["success"] for r in results)

    # print("", flush=True)
    # print("=" * 60, flush=True)
    # print(f"[SUMMARY] tasks={len(results)} avg_score={total_score:.4f} "
    #       f"total_steps={total_steps} all_success={all_success}", flush=True)
    # for r in results:
    #     filled = int(r["score"] * 20)
    #     bar = "\u2588" * filled + "\u2591" * (20 - filled)
    #     print(f"  {r['task_id']:<35} [{bar}] {r['score']:.4f} "
    #           f"({r['steps']} steps)", flush=True)
    # print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
