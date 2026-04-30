"""
utils.py — Shared helpers for training + inference.

This file is intentionally import-safe in a Hugging Face Space:
- no side effects on import
- no OpenAI client dependencies
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# System prompt (shared)
# ---------------------------------------------------------------------------

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
# Local generation (shared; NO external API)
# ---------------------------------------------------------------------------

def generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Run local inference via model.generate() and return decoded completion only."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            max_length=None,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Robust JSON parsing (shared)
# ---------------------------------------------------------------------------

def parse_output(
    raw: str,
    affected: List[str],
    belief_candidates: Optional[List[str]] = None,
) -> Tuple[str, str, Dict[str, float], bool]:
    """Extract (action, reasoning, belief, parse_ok) from raw model output."""
    first = affected[0] if affected else "auth-service"
    fallback = f"investigate {first}"
    candidates = belief_candidates or affected
    n = max(len(candidates), 1)
    uniform = {s: round(1.0 / n, 4) for s in candidates}

    text = re.sub(r"^```[\w]*\n?", "", raw.strip())
    text = re.sub(r"\n?```$", "", text).strip()

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = None

    if not parsed:
        return fallback, "", uniform, False

    thought = str(parsed.get("thought", "")).strip()
    action = str(parsed.get("action", "")).strip() or fallback
    belief = parsed.get("belief", {})

    if not isinstance(belief, dict):
        belief = uniform
    else:
        belief = {
            str(k): float(v)
            for k, v in belief.items()
            if isinstance(v, (int, float))
        }
        belief = {k: max(0.0, belief.get(k, 0.0)) for k in candidates}
        total = sum(belief.values())
        if total > 0:
            belief = {k: v / total for k, v in belief.items()}
        else:
            belief = uniform

    reasoning = f"Thought: {thought}\nBelief: {json.dumps(belief)}"
    return action, reasoning, belief, True


# ---------------------------------------------------------------------------
# Stateful delta-prompt helper (shared)
# ---------------------------------------------------------------------------

def format_stateful_prompt(
    obs: Dict[str, Any],
    last_belief: Dict[str, float],
    last_action: str,
) -> str:
    affected = obs.get("affected_services", [])
    affected_str = ", ".join(affected) if affected else "(see logs)"
    feedback = obs.get("feedback", "") or ""
    task_id = obs.get("task_id", "")
    severity = obs.get("severity", "")
    sla = obs.get("sla_remaining", "?")

    logs_text = "\n".join(f"  {ln}" for ln in obs.get("logs", [])) or "  (no new logs)"
    metrics_text = "\n".join(
        f"  {m.get('service','?')}: error={m.get('error_rate_pct','?')}%  "
        f"latency={m.get('latency_p99_ms','?')}ms  [{m.get('status','?')}]"
        for m in obs.get("metrics", [])
        if isinstance(m, dict)
    ) or "  (no metrics)"

    belief_str = json.dumps(last_belief, indent=2) if last_belief else "{}"
    belief_candidates = obs.get("fan_in_candidates") or obs.get("hypotheses") or affected
    candidate_list = ", ".join(belief_candidates) if belief_candidates else "(none)"
    belief_keys = ",  ".join(f'"{s}": <probability>' for s in belief_candidates)

    return f"""\
Task: {task_id}  |  Severity: {severity}  |  SLA remaining: {sla} min
Affected services: {affected_str}

--- PREVIOUS BELIEF ---
{belief_str}

--- RESULT OF LAST ACTION: {last_action or '(first step)'} ---
{feedback or '(no feedback yet)'}

--- NEW LOGS ---
{logs_text}

--- LIVE METRICS ---
{metrics_text}

--- UPDATE YOUR BELIEF AND CHOOSE AN ACTION ---

Using the new evidence above:
1. Which service is MOST likely the root cause? Update probabilities accordingly.
2. If you investigated a service and found it healthy, DECREASE its probability significantly.
3. Choose ONE action: investigate <service> | assign to <team> | mitigate: <fix> | resolve

Belief constraints:
- Your belief must ONLY contain these keys: {candidate_list}
- You MUST assign probability to ALL candidates.
- No extra keys are allowed.
- Probabilities must sum to 1.0.

Return ONLY valid JSON:
{{
  "thought": "evidence analysis citing specific log lines",
  "belief": {{{belief_keys}}},
  "action": "one valid action"
}}

If belief incorrect -> R2 reward reduced.  If action incorrect -> R1 reward reduced.
Now produce your next step.
"""

