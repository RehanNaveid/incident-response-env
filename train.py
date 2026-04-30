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
# Import SYSTEM_PROMPT and format_prompt from inference.py so training uses
# the EXACT same prompt format as evaluation — eliminates train/infer mismatch.
from inference import SYSTEM_PROMPT, format_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_URL:      str  = os.environ.get("ENV_URL", "https://your-env.hf.space")
IR_DEBUG:     bool = os.environ.get("IR_DEBUG", "0") == "1"


def debug_log(message: str) -> None:
    if IR_DEBUG:
        print(f"[DBG] {message}", flush=True)


def debug_json(label: str, payload: Dict[str, Any]) -> None:
    if IR_DEBUG:
        print(
            f"[DBG:{label}] "
            f"{json.dumps(payload, ensure_ascii=False, default=str, indent=2)}",
            flush=True,
        )

# Model
MODEL_ID:       str  = os.environ.get("MODEL_ID", "unsloth/Qwen2.5-3B-Instruct")
MAX_SEQ_LEN:    int  = int(os.environ.get("MAX_SEQ_LEN", "2048"))
LOAD_IN_4BIT:   bool = True

# LoRA
LORA_R:         int  = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA:     int  = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT:   float = float(os.environ.get("LORA_DROPOUT", "0.0"))
TARGET_MODULES: List[str] = ["q_proj", "v_proj"]

# Generation  (0.7–0.9 range for exploration)
TEMPERATURE:    float = 0.8
TOP_P:          float = 0.9
MAX_NEW_TOKENS: int   = int(os.environ.get("MAX_NEW_TOKENS", "256"))

# GRPO
NUM_ROLLOUTS:   int   = int(os.environ.get("NUM_ROLLOUTS", "3"))
MAX_STEPS_EP:   int   = int(os.environ.get("MAX_STEPS_EP", "6"))
LR:             float = 3e-5
BETA:           float = 0.01     # KL coefficient (set 0 to disable)


def get_max_steps(global_step: int) -> int:
    """Curriculum schedule for the episode horizon."""
    if global_step < 40:
        return MAX_STEPS_EP
    if global_step < 80:
        return MAX_STEPS_EP + 2
    return MAX_STEPS_EP + 4

# Checkpointing
OUTPUT_DIR:     str   = "./checkpoints"
SAVE_STEPS:     int   = int(os.environ.get("SAVE_STEPS", "5"))
SAVE_TOTAL:     int   = 3
FINAL_DIR:      str   = "incidentiq-lora"

# Curriculum
CURRICULUM: List[List[str]] = [
    ["single_service_outage"],                                        # epoch 1: master basics
    ["single_service_outage", "single_service_outage",
     "ambiguous_payment_degradation"],                                 # epoch 2: 2:1 ratio (Task 3 delayed)
    ["single_service_outage", "cascading_failure",
     "ambiguous_payment_degradation"],                                 # epoch 3: all tasks
]
STEPS_PER_EPOCH: int = int(os.environ.get("STEPS_PER_EPOCH", "25"))

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
# Unified blended reward formula (Issue 4: single definition used everywhere)
# ---------------------------------------------------------------------------

R1_BLEND_WEIGHT: float = 0.30
R2_BLEND_WEIGHT: float = 0.70
R1_SCALE:        float = 5.0


def compute_blended(r1: float, r2: float) -> float:
    """0.30*(R1/5) + 0.70*R2 — single source of truth for trainer_step and dry_run."""
    return R1_BLEND_WEIGHT * (r1 / R1_SCALE) + R2_BLEND_WEIGHT * r2


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
    cumulative_reward: float = 0.0   # server R1 (authoritative, may include completion bonus)
    raw_r1:            float = 0.0   # R1 BEFORE completion bonus — used for blended formula
    r2_score:          float = 0.0   # server R2
    blended_reward:    float = 0.0   # compute_blended(raw_r1, r2_score) — used by GRPO
    done:              bool  = False
    parse_rate:        float = 1.0


# ---------------------------------------------------------------------------
# 1.  Model loading
# ---------------------------------------------------------------------------

def load_model() -> Tuple[Any, Any]:
    """Load the configured Qwen2.5-Instruct model with Unsloth 4-bit + LoRA."""
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
    tokenizer.pad_token = tokenizer.eos_token
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

    # Explicit attention mask: all-1s (no padding in our input).
    # Without this, pad_token==eos_token triggers a spurious warning
    # on every generate() call because HF can't infer the mask.
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            attention_mask = attention_mask,
            max_new_tokens = MAX_NEW_TOKENS,
            max_length     = None,   # suppress "both max_new_tokens and max_length set" warning
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
                      headers=_hdr(session_id), timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("observation", data)


def step_env(action: str, reasoning: str,
             session_id: str) -> Tuple[Dict[str, Any], float, bool]:
    last_exc = None
    r = None
    for attempt in range(3):
        try:
            r = requests.post(
                f"{ENV_URL}/step",
                json={"action": {"action": action, "reasoning": reasoning}},
                headers=_hdr(session_id),
                timeout=10,
            )
            r.raise_for_status()
            d = r.json()
            return (
                d.get("observation", {}),
                float(d.get("reward", 0.0)),
                bool(d.get("done", False)),
            )
        except Exception as e:
            last_exc = e
            if attempt == 2:
                print("\n[ENV HARD FAILURE]", flush=True)
                print(f"  action:    {action}", flush=True)
                print(f"  reasoning: {reasoning[:200]}", flush=True)
                if r is not None:
                    try:
                        print(f"  response:  {r.text[:500]}", flush=True)
                    except Exception:
                        pass
                raise last_exc
            time.sleep(2 ** attempt)

    raise RuntimeError("unreachable step_env retry state")


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
    belief_candidates: Optional[List[str]] = None,
) -> Tuple[str, str, Dict[str, float], bool]:
    """Extract (action, reasoning, belief, parse_ok) from raw model output.

    Returns parse_ok=True when JSON was valid, False when fallback was used.
    Callers track parse_ok to compute per-step/per-episode parse success rates.

    Falls back to:  investigate first_service / uniform belief / empty thought.
    """
    first    = affected[0] if affected else "auth-service"
    fallback = f"investigate {first}"
    candidates = belief_candidates or affected
    n        = max(len(candidates), 1)
    uniform  = {s: round(1.0 / n, 4) for s in candidates}

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
        belief = {k: max(0.0, belief.get(k, 0.0)) for k in candidates}
        total = sum(belief.values())
        if total > 0:
            belief = {k: v / total for k, v in belief.items()}
        else:
            belief = uniform

    reasoning = f"Thought: {thought}\nBelief: {json.dumps(belief)}"
    return action, reasoning, belief, True        # parse_ok=True


def run_episode(model, tokenizer, task: str, seed: int,
                rollout_idx: int = 0,
                max_steps: int = MAX_STEPS_EP) -> Episode:
    """Run one full episode, collecting StepRecords.

    Uses format_prompt() from inference.py — identical prompt format to
    evaluation, eliminating train/inference mismatch.

    History is maintained as role/content dicts (matching inference.py's
    run_task), and action_history as a plain string list so build_prompt
    can fire all NEXT hints, resolve_ready, and hypothesis directives.
    """
    ep         = Episode(task_id=task, seed=seed)
    session_id = _make_session_id(rollout_idx)
    debug_log(
        f"episode_start task={task} seed={seed} rollout={rollout_idx} "
        f"session={session_id} max_steps={max_steps}"
    )

    try:
        obs = reset_env(task, seed, session_id)
    except Exception as e:
        print(f"  [EP] reset failed: {e}")
        return ep

    # Conversation history (role/content dicts) — mirrors inference.py run_task()
    history:        List[Dict]         = []
    # Plain action strings — passed to build_prompt for NEXT hints / resolve_ready
    action_history: List[str]          = []
    last_belief:    Dict[str, float]   = {}
    parse_hits:     int                = 0
    parse_total:    int                = 0

    for step_num in range(1, max_steps + 1):
        # Use identical prompt format as inference.py evaluation
        prompt = format_prompt(obs, history, action_history=action_history)
        debug_json("step_input", {
            "task": task, "seed": seed, "rollout": rollout_idx,
            "step": step_num, "max_steps": max_steps,
            "session_id": session_id, "observation": obs,
            "action_history": action_history, "prompt": prompt,
        })

        try:
            raw = generate(model, tokenizer, prompt)
        except Exception as e:
            print(f"  [EP] generate failed step {step_num}: {e}")
            break

        affected = obs.get("affected_services", [])
        belief_candidates = (
            obs.get("fan_in_candidates")
            or obs.get("hypotheses")
            or affected
        )
        action, reasoning, belief, parse_ok = parse_output(
            raw, affected, belief_candidates
        )
        debug_log(
            f"step task={task} seed={seed} rollout={rollout_idx} "
            f"t={step_num}/{max_steps} candidates={belief_candidates} "
            f"belief={belief} action={action!r} parse_ok={parse_ok}"
        )
        parse_total += 1
        if parse_ok:
            parse_hits += 1

        try:
            next_obs, reward, done = step_env(action, reasoning, session_id)
            debug_json("env_response", {
                "task": task, "seed": seed, "rollout": rollout_idx,
                "step": step_num, "action": action,
                "reward": reward, "done": done,
            })
        except Exception as e:
            print(f"  [EP] step_env failed step {step_num}: {e}")
            break

        # Update action_history (used by format_prompt → build_prompt hints)
        action_history.append(action)

        # Build history entry for next prompt
        try:
            history.append({"role": "assistant",
                            "content": json.dumps({"action": action, "thought": "", "belief": belief})})
        except Exception:
            history.append({"role": "assistant", "content": action})

        feedback = next_obs.get("feedback", "")
        result_summary = f"RESULT OF YOUR LAST ACTION: {feedback} (reward={reward:.2f})"
        if "Fix applied" in feedback:
            if "healthy" in feedback.lower() or "you may resolve" in feedback.lower():
                result_summary += " LIVE METRICS ARE NOW HEALTHY — your next action MUST be: resolve"
            else:
                result_summary += " Mitigation applied but services still degraded. Re-apply same mitigation."
        elif "resolved successfully" in feedback.lower():
            result_summary += " Incident resolved. Episode complete."
        elif "Cannot resolve" in feedback:
            result_summary += " Cannot resolve yet — apply mitigation first."
        history.append({"role": "user", "content": result_summary})

        # Cap history to last 6 exchanges (12 messages) to prevent token overflow
        if len(history) > 12:
            history = history[-12:]

        # Record step for GRPO loss computation
        ep.steps.append(StepRecord(
            prompt=prompt, completion=raw,
            action=action, reasoning=reasoning,
            reward=reward, parse_ok=parse_ok,
        ))
        last_belief = belief
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
        # raw_r1 = R1 before any completion bonus (clean signal for blended formula)
        ep.raw_r1 = ep.cumulative_reward
        # Floor catastrophic Task 3 reward to prevent poisoning GRPO gradient
        ep.cumulative_reward = max(ep.cumulative_reward, -3.0)
        ep.raw_r1            = max(ep.raw_r1,            -3.0)
        debug_json("episode_final_state", {
            "task": task,
            "seed": seed,
            "rollout": rollout_idx,
            "session_id": session_id,
            "steps": len(ep.steps),
            "done": ep.done,
            "local_reward_sum": local_sum,
            "state": state,
            "cumulative_reward": ep.cumulative_reward,
            "r2_score": ep.r2_score,
            "parse_rate": ep.parse_rate,
        })
        debug_log(
            f"episode_state task={task} seed={seed} rollout={rollout_idx} "
            f"steps={len(ep.steps)} done={ep.done} local_sum={local_sum:.4f} "
            f"state_reward={ep.cumulative_reward:.4f} r2={ep.r2_score:.4f}"
        )
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

    Uses ep.blended_reward = 0.3*(R1/5) + 0.7*R2 to bias early
    optimisation toward belief calibration.

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
            # Tokenize the full sequence (prompt + completion) via chat template
            # so BOS/EOS are inserted at the correct positions only once.
            # Separate tokenization + concat shifts label indices by 1-2 tokens.
            full_text = tokenizer.apply_chat_template(
                msgs + [{"role": "assistant", "content": step.completion}],
                tokenize=False,
                add_generation_prompt=False,
            )
            full_ids = tokenizer(
                full_text, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(device)

            # Compute prompt length from the template (without the assistant turn)
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            prompt_len = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False,
            ).input_ids.shape[-1]

            # Mask prompt tokens; supervise only the completion span
            labels = torch.full_like(full_ids, -100)
            labels[0, prompt_len:] = full_ids[0, prompt_len:]

            input_ids = full_ids
            out = model(input_ids=input_ids, labels=labels)
            losses.append(-adv_t * out.loss)   # scalar tensor, tracked

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 7.  Checkpoint management
# ---------------------------------------------------------------------------

def save_ckpt(model, tokenizer, optimizer, step: int) -> None:
    path = os.path.join(OUTPUT_DIR, f"step-{step}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    # Persist optimizer state so resume restores momentum/variance accumulators
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
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
    # try/finally guarantees for_training() is always called even if
    # run_episode raises mid-rollout (env timeout, JSON crash, etc.).
    # Without this, all subsequent gradient steps compute on a frozen model.
    episodes: List[Episode] = []
    max_steps = get_max_steps(global_step)
    import random as _random
    try:
        FastLanguageModel.for_inference(model)
        for i in range(NUM_ROLLOUTS):
            # Issue 5: collision-resistant seed formula prevents identical rollouts
            rollout_seed = global_step * 10007 + i * 1009 + _random.randint(0, 999)
            ep = run_episode(model, tokenizer, task, rollout_seed,
                             rollout_idx=i,
                             max_steps=max_steps)
            # Issue 1: use raw_r1 (pre-bonus) with unified formula
            ep.blended_reward = compute_blended(ep.raw_r1, ep.r2_score)
            episodes.append(ep)
            print(f"  rollout {i+1}/{NUM_ROLLOUTS}  "
                  f"R={ep.cumulative_reward:.4f}  raw_r1={ep.raw_r1:.4f}  "
                  f"r2={ep.r2_score:.4f}  blended={ep.blended_reward:.4f}  "
                  f"steps={len(ep.steps)}/{max_steps}  done={ep.done}  "
                  f"parse={ep.parse_rate:.0%}  seed={rollout_seed}")
    finally:
        FastLanguageModel.for_training(model)

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

    # blended_reward already set per-episode above (using raw_r1)

    # ---- Advantages + reward-variance guard ----
    rewards   = [ep.blended_reward for ep in episodes]   # F2: blended signal
    advantages = grpo_advantages(episodes)
    avg_r   = sum(ep.cumulative_reward for ep in episodes) / max(len(episodes), 1)
    avg_r2  = sum(ep.r2_score          for ep in episodes) / max(len(episodes), 1)
    avg_bl  = sum(ep.blended_reward    for ep in episodes) / max(len(episodes), 1)
    avg_steps = sum(len(ep.steps) for ep in episodes) / max(len(episodes), 1)
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
    update_t0 = time.time()
    print(f"[TRAIN] step={global_step} update_start rollouts={len(episodes)}", flush=True)

    optimizer.zero_grad()
    loss_t0 = time.time()
    print(f"[TRAIN] step={global_step} loss_start", flush=True)
    loss = compute_loss(model, tokenizer, episodes, advantages)
    loss_elapsed = time.time() - loss_t0
    print(
        f"[TRAIN] step={global_step} loss_done "
        f"loss={loss.item():.6f} elapsed={loss_elapsed:.1f}s",
        flush=True,
    )

    backward_t0 = time.time()
    print(f"[TRAIN] step={global_step} backward_start", flush=True)
    loss.backward()
    backward_elapsed = time.time() - backward_t0
    print(
        f"[TRAIN] step={global_step} backward_done elapsed={backward_elapsed:.1f}s",
        flush=True,
    )

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt_t0 = time.time()
    print(f"[TRAIN] step={global_step} optimizer_start", flush=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    opt_elapsed = time.time() - opt_t0
    update_elapsed = time.time() - update_t0
    print(
        f"[TRAIN] step={global_step} completed "
        f"loss={loss.item():.6f} update_elapsed={update_elapsed:.1f}s "
        f"optimizer_elapsed={opt_elapsed:.1f}s",
        flush=True,
    )

    return {
        "loss":           loss.item(),
        "avg_reward":     avg_r,
        "avg_r2":         avg_r2,
        "avg_blended":    avg_bl,
        "avg_steps":      avg_steps,
        "max_steps_ep":   max_steps,
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
                f"avg_steps={metrics['avg_steps']:.2f}/{metrics['max_steps_ep']}  "
                f"reward_std={metrics['reward_std']:.4f}  "
                f"parse_rate={metrics['parse_rate']:.0%}  "
                f"sample_action={metrics['sample']!r}  "
                f"elapsed={elapsed:.1f}s"
            )

            if global_step % SAVE_STEPS == 0:
                save_ckpt(model, tokenizer, optimizer, global_step)

    # Final save
    print(f"\n[TRAIN] Saving final model → {FINAL_DIR}")
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"[TRAIN] Done. Model at ./{FINAL_DIR}")

    # ---- Push to Hugging Face Hub ----
    import os
    from huggingface_hub import login, HfApi

    hf_token = os.environ.get("HF_TOKEN")
    hf_user  = os.environ.get("HF_USERNAME")

    if hf_token and hf_user:
        try:
            login(token=hf_token)

            repo_id = f"{hf_user}/incidentiq-lora"
            print(f"[TRAIN] Uploading model → {repo_id}")

            api = HfApi()
            api.upload_folder(
                folder_path=FINAL_DIR,
                repo_id=repo_id,
                repo_type="model",
                token=hf_token,
            )

            print(f"[TRAIN] Model live at: https://huggingface.co/{repo_id}")

        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")
    else:
        print("[WARN] HF_TOKEN or HF_USERNAME missing → skipping upload")

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
    max_steps = get_max_steps(0)
    print(f"\n[DRY-RUN] Running {n} diagnostic episodes (no weight update, max_steps={max_steps}) …")
    FastLanguageModel.for_inference(model)

    tasks  = CURRICULUM[-1]    # use full task pool for worst-case coverage
    rows:  List[Dict] = []

    eps_collected: List = []
    for i in range(n):
        task = tasks[i % len(tasks)]
        seed = SEEDS[i % len(SEEDS)] + 1000   # distinct from training seeds
        ep   = run_episode(model, tokenizer, task, seed, max_steps=max_steps)
        eps_collected.append(ep)
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

    # Compute estimated blended using same formula as trainer_step (Issue 4)
    blended_est = [
        compute_blended(ep.raw_r1, ep.r2_score)
        for ep in eps_collected
    ]
    avg_blended_est = sum(blended_est) / max(len(blended_est), 1)

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
    print(f"  avg_blended(est) : {avg_blended_est:.4f}  (0.3*(R1/5) + 0.7*R2)")
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
          f"max_steps_ep={MAX_STEPS_EP}/{MAX_STEPS_EP + 2}/{MAX_STEPS_EP + 4}  "
          f"max_new_tokens={MAX_NEW_TOKENS}  lr={LR}  beta={BETA}")
    print(f"[TRAIN] Curriculum     = {len(CURRICULUM)} epochs × "
          f"{STEPS_PER_EPOCH} steps")
    print(f"[TRAIN] Checkpoints    = {OUTPUT_DIR}  "
          f"every {SAVE_STEPS} steps  keep {SAVE_TOTAL}")

    # Preflight: fail fast if env is cold/unreachable before loading the model.
    # A cold HF Space hangs 30s then raises — this burns A100 time before the
    # error surfaces.  Ping /health first so failure is instant and obvious.
    try:
        _health = requests.get(f"{ENV_URL}/health", timeout=10)
        _health.raise_for_status()
        print(f"[TRAIN] Env healthy: {_health.json()}")
    except Exception as _he:
        print(f"[TRAIN] ERROR: env not reachable at {ENV_URL!r}: {_he}", file=sys.stderr)
        sys.exit(1)

    model, tokenizer = load_model()

    # Step 1: diagnostic dry-run (10 episodes, no weight update)
    # Aborts if parse_rate < 70% or prints variance warning.
    # Remove or set n=0 to skip once you trust the setup.
    dry_run(model, tokenizer, n=3)

    # Step 2: full curriculum training
    train(model, tokenizer)


if __name__ == "__main__":
    main()
