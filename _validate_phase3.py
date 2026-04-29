"""Phase 3 prompt structure validation — no server required."""
import sys, inspect
sys.path.insert(0, ".")

FAIL = []
PASS = 0

def check(cond, msg):
    global PASS
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL.append(msg)
        print(f"  FAIL  {msg}")

import inference

check(callable(inference.format_prompt), "format_prompt is callable")
check(callable(inference.ask_llm),       "ask_llm is callable")

# Temperature
ask_src = inspect.getsource(inference.ask_llm)
check("temperature=0.8" in ask_src, "temperature=0.8 in ask_llm")
check("format_prompt(" in ask_src,   "ask_llm calls format_prompt")
check("user_prompt" not in ask_src,  "old user_prompt variable removed from ask_llm")

# format_prompt calls build_prompt
fp_src = inspect.getsource(inference.format_prompt)
check("build_prompt" in fp_src, "format_prompt calls build_prompt")

# Render with mock observation
mock_obs = {
    "affected_services": ["auth-service", "db-primary"],
    "task_id": "cascading_failure",
    "feedback": "",
    "incident_description": "auth latency spike",
    "logs": ["[CRIT] auth-service error_rate=35%"],
    "metrics": [],
    "severity": "high",
    "sla_remaining": 10,
    "team_roster": {"sre-backend": "available"},
}
result = inference.format_prompt(mock_obs, [])

check("CURRENT OBSERVATION"  in result, "Section: CURRENT OBSERVATION")
check("YOUR PREVIOUS STEPS"  in result, "Section: YOUR PREVIOUS STEPS")
check("INSTRUCTIONS"         in result, "Section: INSTRUCTIONS")
check("OUTPUT FORMAT"        in result, "Section: OUTPUT FORMAT (STRICT)")
check("CRITICAL CONSTRAINTS" in result, "Section: CRITICAL CONSTRAINTS")
check("auth-service"         in result, "affected service name in prompt")
check("belief"               in result, "belief keyword present")
check("R2 reward"            in result, "R2 reward wording present")
check("R1 reward"            in result, "R1 reward wording present")
check("probability MUST decrease significantly" in result,
      "causal constraint: healthy -> decrease probability")
check('first step' in result or "no history" in result,
      "history section shows first-step message when history is empty")

# Belief template uses actual service names
check('"auth-service": <probability>' in result, "belief template: auth-service key")
check('"db-primary": <probability>'   in result, "belief template: db-primary key")

# SYSTEM_PROMPT RL alignment
sp = inference.SYSTEM_PROMPT
check("reinforcement learning" in sp.lower(), "SYSTEM_PROMPT mentions reinforcement learning")
check("R1" in sp and "R2" in sp,              "SYSTEM_PROMPT mentions R1 and R2")
check("0.60" in sp and "0.40" in sp,          "SYSTEM_PROMPT states R1/R2 weights (0.60/0.40)")
check("belief" in sp.lower(),                 "SYSTEM_PROMPT mentions belief updates")

# Weights
check(inference.R1_WEIGHT == 0.60, f"R1_WEIGHT=0.60 (got {inference.R1_WEIGHT})")
check(inference.R2_WEIGHT == 0.40, f"R2_WEIGHT=0.40 (got {inference.R2_WEIGHT})")

# Check history rendering
mock_history = [
    {"role": "user",      "content": "You are step 1"},
    {"role": "assistant", "content": '{"thought":"t","belief":{},"action":"investigate auth-service"}'},
]
result_with_history = inference.format_prompt(mock_obs, mock_history)
check("[ENV]" in result_with_history, "history renders [ENV] lines for user messages")
check("[YOU]" in result_with_history, "history renders [YOU] lines for assistant messages")

print()
total = PASS + len(FAIL)
if FAIL:
    print(f"FAILED {len(FAIL)}/{total} checks:")
    for f in FAIL:
        print(f"  * {f}")
    sys.exit(1)
else:
    print(f"All {PASS} Phase 3 prompt checks PASS")
