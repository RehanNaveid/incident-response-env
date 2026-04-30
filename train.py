"""
train.py — Phase 3 GRPO RL training for Incident Response agent.

Local model : unsloth/Qwen2.5-7B-Instruct  (4-bit + LoRA)
Environment : Incident Response env (HTTP, already deployed)
Algorithm   : GRPO  (Group Relative Policy Optimization)
Curriculum  : single_service_outage → cascading_failure → all tasks

NO cloud / external LLM API calls anywhere in this file.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Unsloth  (must be installed: pip install unsloth)
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel

# ---------------------------------------------------------------------------
# TRL — used for GRPOConfig (hyperparameter container)
# ---------------------------------------------------------------------------
from trl import GRPOConfig

# ---------------------------------------------------------------------------
# Reuse existing Phase 3 prompt infrastructure from inference.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import format_prompt, SYSTEM_PROMPT  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_URL:      str  = os.environ.get("ENV_URL", "https://your-env.hf.space")

# Model
MODEL_ID:       str  = "unsloth/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN:    int  = 4096
LOAD_IN_4BIT:   bool = True

# LoRA
LORA_R:         int  = 16
LORA_ALPHA:     int  = 32
LORA_DROPOUT:   float = 0.05
TARGET_MODULES: List[str] = ["q_proj", "v_proj"]

# Generation  (0.7–0.9 range for exploration)
TEMPERATURE:    float = 0.8
TOP_P:          float = 0.9
MAX_NEW_TOKENS: int   = 320   # F5: enough for JSON belief; ~20% cheaper than 400

# GRPO
NUM_ROLLOUTS:   int   = 6        # rollouts per training step (6–8)
MAX_STEPS_EP:   int   = 8        # hard budget per episode (6–10)
LR:             float = 3e-5
BETA:           float = 0.01     # KL coefficient (set 0 to disable)

# Checkpointing
OUTPUT_DIR:     str   = "./checkpoints"
SAVE_STEPS:     int   = 25
SAVE_TOTAL:     int   = 3
FINAL_DIR:      str   = "incidentiq-lora"

# Curriculum
CURRICULUM: List[List[str]] = [
    ["single_service_outage"],
    ["single_service_outage", "cascading_failure"],
    ["single_service_outage", "cascading_failure", "ambiguous_payment_degradation"],
]
STEPS_PER_EPOCH: int = 50

SEEDS: List[int] = [42, 7, 13, 99, 2024, 314]

# ---------------------------------------------------------------------------
# GRPOConfig  (used for config container + future trainer.train() integration)
# ---------------------------------------------------------------------------
grpo_config = GRPOConfig(
    output_dir        = OUTPUT_DIR,
    save_steps        = SAVE_STEPS,
    save_total_limit  = SAVE_TOTAL,
    learning_rate     = LR,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2,
    max_completion_length       = MAX_NEW_TOKENS,
    num_generations             = NUM_ROLLOUTS,
    logging_steps               = 1,
    report_to                   = "none",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    prompt:     str
    completion: str
    action:     str
    reasoning:  str
    reward:     float = 0.0
    parse_ok:   bool  = True    # True if JSON parsed cleanly; False if fallback used


@dataclass
class Episode:
    task_id:           str
    seed:              int
    steps:             List[StepRecord] = field(default_factory=list)
    cumulative_reward: float = 0.0   # server R1 (authoritative)
    r2_score:          float = 0.0   # server R2
    blended_reward:    float = 0.0   # F2: 0.5*(R1/5) + 0.5*R2  — used by GRPO
    done:              bool  = False
    parse_rate:        float = 1.0


# ---------------------------------------------------------------------------
# 1.  Model loading
# ---------------------------------------------------------------------------

def load_model() -> Tuple[Any, Any]:
    """Load Qwen2.5-7B-Instruct with Unsloth 4-bit + LoRA."""
    print(f"[TRAIN] Loading {MODEL_ID} (4-bit) …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_ID,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit   = LOAD_IN_4BIT,
        dtype          = None,          # auto (bfloat16 on A100)
    )
    print("[TRAIN] Applying LoRA r=16 alpha=32 …")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_R,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = LORA_DROPOUT,
        target_modules             = TARGET_MODULES,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
    )
    print("[TRAIN] Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2.  Local generation  (NO external API)
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str) -> str:
    """Run local inference via model.generate().

    Temperature 0.8 ensures exploration for RL rollouts.
    Returns raw decoded completion string only (prompt stripped).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize              = True,
        add_generation_prompt = True,
        return_tensors        = "pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = TEMPERATURE,
            top_p          = TOP_P,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )

    new_tokens = out_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 3.  Environment interaction
# ---------------------------------------------------------------------------

def _make_session_id(rollout_idx: int = 0) -> str:
    """F3: unique session per rollout — prevents episode state bleed."""
    ts = int(time.time() * 1000) % 10_000_000
    return f"train-{os.getpid()}-{ts}-{rollout_idx}"


def _hdr(session_id: str) -> Dict[str, str]:
    return {"X-Session-Id": session_id}


def reset_env(task: str, seed: int, session_id: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_id": task, "seed": seed},
                      headers=_hdr(session_id), timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("observation", data)


def step_env(action: str, reasoning: str,
             session_id: str) -> Tuple[Dict[str, Any], float, bool]:
    r = requests.post(f"{ENV_URL}/step",
                      json={"action": {"action": action, "reasoning": reasoning}},
                      headers=_hdr(session_id), timeout=30)
    r.raise_for_status()
    d = r.json()
    return d.get("observation", {}), float(d.get("reward", 0.0)), bool(d.get("done", False))


def get_state(session_id: str) -> Dict[str, Any]:
    r = requests.get(f"{ENV_URL}/state", headers=_hdr(session_id), timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# 4.  Robust JSON parsing  (model output may be messy early in training)
# ---------------------------------------------------------------------------

def parse_output(
    raw: str,
    affected: List[str],
) -> Tuple[str, str, Dict[str, float], bool]:
    """Extract (action, reasoning, belief, parse_ok) from raw model output.

    Returns parse_ok=True when JSON was valid, False when fallback was used.
    Callers track parse_ok to compute per-step/per-episode parse success rates.

    Falls back to:  investigate first_service / uniform belief / empty thought.
    """
    first    = affected[0] if affected else "auth-service"
    fallback = f"investigate {first}"
    n        = max(len(affected), 1)
    uniform  = {s: round(1.0 / n, 4) for s in affected}

    # Strip markdown fences
    text = re.sub(r"^```[\w]*\n?", "", raw.strip())
    text = re.sub(r"\n?```$", "", text).strip()

    parsed: Optional[Dict] = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if not parsed:
        return fallback, "", uniform, False      # parse_ok=False

    thought = str(parsed.get("thought", "")).strip()
    action  = str(parsed.get("action",  "")).strip() or fallback
    belief  = parsed.get("belief", {})

    if not isinstance(belief, dict):
        belief = uniform
    else:
        belief = {str(k): float(v) for k, v in belief.items()
                  if isinstance(v, (int, float))}

    reasoning = f"Thought: {thought}\nBelief: {json.dumps(belief)}"
    return action, reasoning, belief, True        # parse_ok=True


# ---------------------------------------------------------------------------
# 5.  Rollout — one complete episode
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5b.  Stateful training prompt  (Fix 6)
# ---------------------------------------------------------------------------

def format_stateful_prompt(
    obs:         Dict[str, Any],
    last_belief: Dict[str, float],
    last_action: str,
) -> str:
    """Focused POMDP-style delta prompt used during RL training rollouts.

    Instead of re-feeding full history (which duplicates feedback and creates
    noise), shows only the minimal state delta the agent needs to update belief:
      1. Previous belief  — what it believed before this step
      2. Result/feedback  — outcome of last action
      3. New logs/metrics — fresh evidence
      4. Update directive — explicit instruction to revise belief then act

    This forms a proper belief-update loop (POMDP filter), not stateless
    guessing.  history duplication eliminated → cleaner belief gradient.
    """
    affected     = obs.get("affected_services", [])
    affected_str = ", ".join(affected) if affected else "(see logs)"
    feedback     = obs.get("feedback", "") or ""
    task_id      = obs.get("task_id", "")
    severity     = obs.get("severity", "")
    sla          = obs.get("sla_remaining", "?")

    logs_text = "\n".join(
        f"  {ln}" for ln in obs.get("logs", [])
    ) or "  (no new logs)"

    metrics_text = "\n".join(
        f"  {m.get('service','?')}: error={m.get('error_rate_pct','?')}%  "
        f"latency={m.get('latency_p99_ms','?')}ms  [{m.get('status','?')}]"
        for m in obs.get("metrics", [])
        if isinstance(m, dict)
    ) or "  (no metrics)"

    belief_str  = json.dumps(last_belief, indent=2) if last_belief else "{}"
    belief_keys = ",  ".join(f'"{s}": <probability>' for s in affected)

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

Return ONLY valid JSON:
{{
  "thought": "evidence analysis citing specific log lines",
  "belief": {{{belief_keys}}},
  "action": "one valid action"
}}

If belief incorrect -> R2 reward reduced.  If action incorrect -> R1 reward reduced.
Now produce your next step.
"""


def run_episode(model, tokenizer, task: str, seed: int,
                rollout_idx: int = 0) -> Episode:
    """Run one full episode, collecting StepRecords.

    Fix 3: unique session_id per rollout (pid + timestamp + rollout_idx).
    Fix 4: /state cumulative_reward is authoritative; local sum is fallback.
    Fix 6: uses format_stateful_prompt() for focused belief-update context.
    """
    ep         = Episode(task_id=task, seed=seed)
    session_id = _make_session_id(rollout_idx)

    try:
        obs = reset_env(task, seed, session_id)
    except Exception as e:
        print(f"  [EP] reset failed: {e}")
        return ep

    last_belief: Dict[str, float] = {}
    last_action: str              = ""
    parse_hits:  int              = 0
    parse_total: int              = 0

    for step_num in range(1, MAX_STEPS_EP + 1):
        # F6: stateful delta-prompt instead of full history replay
        prompt = format_stateful_prompt(obs, last_belief, last_action)

        try:
            raw = generate(model, tokenizer, prompt)
        except Exception as e:
            print(f"  [EP] generate failed step {step_num}: {e}")
            break

        affected = obs.get("affected_services", [])
        action, reasoning, belief, parse_ok = parse_output(raw, affected)
        parse_total += 1
        if parse_ok:
            parse_hits += 1

        try:
            next_obs, reward, done = step_env(action, reasoning, session_id)
        except Exception as e:
            print(f"  [EP] step_env failed step {step_num}: {e}")
            break

        ep.steps.append(StepRecord(
            prompt=prompt, completion=raw,
            action=action, reasoning=reasoning,
            reward=reward, parse_ok=parse_ok,
        ))
        last_belief = belief
        last_action = action
        obs         = next_obs

        if done:
            ep.done = True
            break

    ep.parse_rate = parse_hits / max(parse_total, 1)

    # F4: /state is authoritative (includes terminal R1 bonus + R2 trajectory)
    local_sum = sum(s.reward for s in ep.steps)
    try:
        state = get_state(session_id)
        ep.cumulative_reward = float(state.get("cumulative_reward") or local_sum)
        ep.r2_score          = float(state.get("info", {}).get("r2_score", 0.0) or 0.0)
    except Exception as e:
        print(f"  [EP] get_state failed: {e}")
        ep.cumulative_reward = local_sum

    # NOTE: blended_reward is computed AFTER all episodes in the rollout batch
    # so that per-batch R1 normalisation (Fix 3) has access to all R1 values.
    # See trainer_step() where ep.blended_reward is set for each episode.

    return ep


# ---------------------------------------------------------------------------
# 6.  GRPO reward normalisation + policy-gradient loss
# ---------------------------------------------------------------------------

def grpo_advantages(episodes: List[Episode]) -> List[float]:
    """Normalise blended episode rewards within the group (GRPO §3.1).

    Uses ep.blended_reward = 0.5*(R1/5) + 0.5*R2 so both task accuracy
    and belief calibration contribute equally to the policy gradient.

    advantage_i = (blended_i - mean) / (std + eps)
    """
    rewards = [ep.blended_reward for ep in episodes]
    mu      = sum(rewards) / max(len(rewards), 1)
    var     = sum((r - mu) ** 2 for r in rewards) / max(len(rewards), 1)
    sigma   = math.sqrt(var)
    eps     = 1e-6
    return [(r - mu) / (sigma + eps) for r in rewards]


def compute_loss(model, tokenizer,
                 episodes: List[Episode],
                 advantages: List[float]) -> torch.Tensor:
    """GRPO policy-gradient loss — memory-safe implementation.

    Collects per-(episode, step) loss tensors into a Python list and calls
    torch.stack().mean() once.  This avoids the 48-node accumulation graph
    that the old in-place total = total + ... pattern creates and never frees
    between rollouts, which causes T4 OOM after ~3 training steps.

    Loss = -mean[ advantage * NLL(completion | prompt) ]
    Only completion tokens are supervised (prompt tokens masked with -100).
    """
    device = next(model.parameters()).device
    losses: List[torch.Tensor] = []

    model.train()
    for ep, adv in zip(episodes, advantages):
        if not ep.steps:
            continue
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)

        for step in ep.steps:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": step.prompt},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            compl_ids = tokenizer(
                step.completion, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(device)

            input_ids = torch.cat([prompt_ids, compl_ids], dim=-1)
            labels    = torch.full_like(input_ids, -100)
            labels[0, prompt_ids.shape[-1]:] = compl_ids[0]

            out = model(input_ids=input_ids, labels=labels)
            losses.append(-adv_t * out.loss)   # scalar tensor, tracked

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 7.  Checkpoint management
# ---------------------------------------------------------------------------

def save_ckpt(model, tokenizer, step: int) -> None:
    path = os.path.join(OUTPUT_DIR, f"step-{step}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[TRAIN] Checkpoint → {path}")

    saved = sorted(
        int(d.split("-")[-1])
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("step-") and d.split("-")[-1].isdigit()
    )
    while len(saved) > SAVE_TOTAL:
        old = saved.pop(0)
        shutil.rmtree(os.path.join(OUTPUT_DIR, f"step-{old}"), ignore_errors=True)
        print(f"[TRAIN] Pruned checkpoint step-{old}")


# ---------------------------------------------------------------------------
# 8.  Training loop  (curriculum + GRPO update per step)
# ---------------------------------------------------------------------------

# Thresholds for training stability guards
_MIN_PARSE_RATE:     float = 0.70   # stop training if parse rate falls below this
_MIN_REWARD_STD:     float = 0.05   # warn if reward std below this (GRPO dead zone)


def trainer_step(model, tokenizer, optimizer,
                 task: str, seed_base: int, global_step: int) -> Dict[str, float]:
    """One GRPO training step.

    1. Switch to inference mode
    2. Generate NUM_ROLLOUTS episodes (sequential on single GPU)
    3. Compute GRPO advantages
    4. Switch to training mode
    5. Backprop policy-gradient loss
    6. Return metrics dict

    Safety guards:
      - Halts training if parse_rate < _MIN_PARSE_RATE (belief never updates)
      - Warns if reward std < _MIN_REWARD_STD (GRPO has no signal)
    """
    # ---- Rollouts ----
    FastLanguageModel.for_inference(model)
    episodes: List[Episode] = []
    for i in range(NUM_ROLLOUTS):
        ep = run_episode(model, tokenizer, task, seed_base + i,
                         rollout_idx=i)   # F3: unique session per rollout
        episodes.append(ep)
        print(f"  rollout {i+1}/{NUM_ROLLOUTS}  "
              f"R={ep.cumulative_reward:.4f}  r2={ep.r2_score:.4f}  "
              f"blended={ep.blended_reward:.4f}  "
              f"steps={len(ep.steps)}  done={ep.done}  "
              f"parse={ep.parse_rate:.0%}")

    # ---- Parse-rate guard ----
    avg_parse = sum(ep.parse_rate for ep in episodes) / max(len(episodes), 1)
    if avg_parse < _MIN_PARSE_RATE:
        raise RuntimeError(
            f"[TRAIN] HALT: parse_rate={avg_parse:.1%} < threshold {_MIN_PARSE_RATE:.0%}.\n"
            f"  Model is producing garbage JSON. Possible causes:\n"
            f"  - TEMPERATURE too high (try 0.7)\n"
            f"  - prompt format mismatch\n"
            f"  - model not following instruction format yet (wait 50+ steps)\n"
            f"  Run dry_run() to diagnose without updating weights."
        )

    # ---- Fix 3: per-batch R1 normalisation (replaces hardcoded /5.0) ----
    # R1 (cumulative_reward) scale varies by task; normalising within the
    # current rollout batch ensures both R1 and R2 have comparable gradient
    # magnitude regardless of task difficulty or episode length.
    r1_vals  = [ep.cumulative_reward for ep in episodes]
    r1_mean  = sum(r1_vals) / max(len(r1_vals), 1)
    r1_std   = math.sqrt(sum((r - r1_mean) ** 2 for r in r1_vals)
                         / max(len(r1_vals), 1)) + 1e-6
    for ep in episodes:
        r1_norm           = (ep.cumulative_reward - r1_mean) / r1_std
        ep.blended_reward = 0.6 * r1_norm + 0.4 * ep.r2_score

    # ---- Advantages + reward-variance guard ----
    rewards   = [ep.blended_reward for ep in episodes]   # F2: blended signal
    advantages = grpo_advantages(episodes)
    avg_r   = sum(ep.cumulative_reward for ep in episodes) / max(len(episodes), 1)
    avg_r2  = sum(ep.r2_score          for ep in episodes) / max(len(episodes), 1)
    avg_bl  = sum(ep.blended_reward    for ep in episodes) / max(len(episodes), 1)
    std_r   = math.sqrt(sum((r - sum(rewards)/max(len(rewards),1)) ** 2
                            for r in rewards) / max(len(rewards), 1))
    best    = max(episodes, key=lambda e: e.cumulative_reward)
    sample  = best.steps[-1].action if best.steps else "N/A"

    if std_r < _MIN_REWARD_STD:
        print(f"[WARN] Low reward variance std={std_r:.4f} < {_MIN_REWARD_STD} — "
              f"GRPO signal is weak. Rewards: {[f'{r:.3f}' for r in rewards]}. "
              f"Consider more diverse seeds or checking env reward range.")

    # ---- Update ----
    FastLanguageModel.for_training(model)
    optimizer.zero_grad()
    loss = compute_loss(model, tokenizer, episodes, advantages)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss":           loss.item(),
        "avg_reward":     avg_r,
        "avg_r2":         avg_r2,
        "avg_blended":    avg_bl,
        "reward_std":     std_r,
        "parse_rate":     avg_parse,
        "sample":         sample,
    }


def train(model, tokenizer) -> None:
    """Main curriculum training loop."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    optimizer    = AdamW(model.parameters(), lr=LR)
    global_step  = 0

    for epoch, task_pool in enumerate(CURRICULUM, 1):
        print(f"\n[TRAIN] ══ Epoch {epoch} | tasks={task_pool} ══")

        for local in range(1, STEPS_PER_EPOCH + 1):
            global_step += 1
            task      = task_pool[(local - 1) % len(task_pool)]
            # Fix 4: prime-multiplier seed avoids periodic cycling.
            # Period of global_step * 7 mod 2^31 >> curriculum length.
            seed_base = SEEDS[0] + global_step * 7

            print(f"\n[TRAIN] step={global_step} task={task} seed_base={seed_base}")
            t0      = time.time()
            metrics = trainer_step(model, tokenizer, optimizer,
                                   task, seed_base, global_step)
            elapsed = time.time() - t0

            print(
                f"[TRAIN] "
                f"loss={metrics['loss']:.6f}  "
                f"avg_reward={metrics['avg_reward']:.4f}  "
                f"avg_r2={metrics['avg_r2']:.4f}  "
                f"avg_blended={metrics['avg_blended']:.4f}  "
                f"reward_std={metrics['reward_std']:.4f}  "
                f"parse_rate={metrics['parse_rate']:.0%}  "
                f"sample_action={metrics['sample']!r}  "
                f"elapsed={elapsed:.1f}s"
            )

            if global_step % SAVE_STEPS == 0:
                save_ckpt(model, tokenizer, global_step)

    # Final save
    print(f"\n[TRAIN] Saving final model → {FINAL_DIR}")
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"[TRAIN] Done. Model at ./{FINAL_DIR}")


# ---------------------------------------------------------------------------
# 9.  Dry-run  — diagnostics WITHOUT weight updates
# ---------------------------------------------------------------------------

def dry_run(model, tokenizer, n: int = 10) -> None:
    """Run N episodes with no weight updates; print diagnostic report.

    Verifies before committing to training:
      - parse_rate >= 70%  (model follows JSON format)
      - reward std >= 0.05 (env produces varied signal for GRPO)
      - R2 varies across episodes (belief calibration is active)

    Call from main() or standalone:
        model, tok = load_model()
        dry_run(model, tok, n=10)
    """
    print(f"\n[DRY-RUN] Running {n} diagnostic episodes (no weight update) …")
    FastLanguageModel.for_inference(model)

    tasks  = CURRICULUM[-1]    # use full task pool for worst-case coverage
    rows:  List[Dict] = []

    for i in range(n):
        task = tasks[i % len(tasks)]
        seed = SEEDS[i % len(SEEDS)] + 1000   # distinct from training seeds
        ep   = run_episode(model, tokenizer, task, seed)
        rows.append({
            "ep":     i + 1,
            "task":   task[:20],
            "reward": ep.cumulative_reward,
            "r2":     ep.r2_score,
            "parse":  ep.parse_rate,
            "steps":  len(ep.steps),
            "done":   ep.done,
        })
        print(f"  ep={i+1:02d}  task={task[:22]:<24}  "
              f"R={ep.cumulative_reward:.4f}  r2={ep.r2_score:.4f}  "
              f"parse={ep.parse_rate:.0%}  steps={len(ep.steps)}")

    # ---- Aggregate diagnostics ----
    rewards     = [r["reward"] for r in rows]
    parse_rates = [r["parse"]  for r in rows]
    r2_scores   = [r["r2"]     for r in rows]

    avg_r    = sum(rewards)     / max(len(rewards),     1)
    avg_p    = sum(parse_rates) / max(len(parse_rates), 1)
    avg_r2   = sum(r2_scores)   / max(len(r2_scores),   1)
    std_r    = math.sqrt(sum((r - avg_r) ** 2 for r in rewards) / max(len(rewards), 1))

    print("\n[DRY-RUN] ─── Diagnostic Summary ───────────────────────────────")
    print(f"  Episodes         : {n}")
    print(f"  avg_reward       : {avg_r:.4f}")
    print(f"  reward_std       : {std_r:.4f}  {'✓ OK' if std_r >= _MIN_REWARD_STD else '✗ LOW — GRPO will be weak'}")
    print(f"  avg_parse_rate   : {avg_p:.1%}  {'✓ OK' if avg_p >= _MIN_PARSE_RATE else '✗ LOW — R2 cannot learn'}")
    print(f"  avg_r2           : {avg_r2:.4f}")
    print("[DRY-RUN] ────────────────────────────────────────────────────────")

    if avg_p < _MIN_PARSE_RATE:
        print("[DRY-RUN] ✗ ABORT: parse_rate below threshold. "
              "Fix prompt/JSON format before training.")
    elif std_r < _MIN_REWARD_STD:
        print("[DRY-RUN] ⚠ WARNING: reward variance is low. "
              "GRPO can still run but convergence will be slow.")
    else:
        print("[DRY-RUN] ✓ READY: system looks healthy. Proceed to train().")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("[TRAIN] Phase 3 — GRPO RL pipeline (local model, no cloud API)")
    print(f"[TRAIN] ENV_URL        = {ENV_URL}")
    print(f"[TRAIN] MODEL_ID       = {MODEL_ID}")
    print(f"[TRAIN] LoRA           = r={LORA_R}  alpha={LORA_ALPHA}  "
          f"targets={TARGET_MODULES}")
    print(f"[TRAIN] GRPO           = rollouts={NUM_ROLLOUTS}  "
          f"max_steps_ep={MAX_STEPS_EP}  lr={LR}  beta={BETA}")
    print(f"[TRAIN] Curriculum     = {len(CURRICULUM)} epochs × "
          f"{STEPS_PER_EPOCH} steps")
    print(f"[TRAIN] Checkpoints    = {OUTPUT_DIR}  "
          f"every {SAVE_STEPS} steps  keep {SAVE_TOTAL}")

    model, tokenizer = load_model()

    # Step 1: diagnostic dry-run (10 episodes, no weight update)
    # Aborts if parse_rate < 70% or prints variance warning.
    # Remove or set n=0 to skip once you trust the setup.
    dry_run(model, tokenizer, n=10)

    # Step 2: full curriculum training
    train(model, tokenizer)


if __name__ == "__main__":
    main()
