"""
inference.py — OpenEnv RL Hackathon submission.
Email Triage Environment: classifies, prioritizes, and acts on inbox emails.
"""

import os
import sys
import json

from openai import OpenAI

# ── Environment variables ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Import environment ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.environment import EmailTriageEnv, Action

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email triage assistant. You will be given a list of emails and must take actions to triage them efficiently.

For each step, respond with a SINGLE JSON action object:
- "email_id": id of the email to act on (string)
- "action": one of "classify", "prioritize", "reply", "archive", "escalate"
- "value": category for classify (spam/work/personal/newsletter), level for prioritize (high/medium/low), text for reply; null for archive/escalate

Strategy:
1. Classify each email by category first.
2. Prioritize each email by urgency.
3. Archive spam and newsletters.
4. Reply to high-priority work emails with a brief professional reply.
5. Escalate critical work issues (server outage, security breach).

Respond ONLY with valid JSON. No explanation, no markdown fences.
Example: {"email_id": "e1", "action": "classify", "value": "work"}
"""


def get_action(obs_dict: dict, history: list):
    obs_text = (
        f"Step {obs_dict['step']} | Inbox size: {obs_dict['inbox_size']}\n"
        f"Message: {obs_dict['message']}\n\n"
        f"Emails:\n{json.dumps(obs_dict['emails'], indent=2)}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-20:])
    messages.append({"role": "user", "content": obs_text})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return json.loads(raw), raw


def run_task(task_name: str) -> None:
    env = EmailTriageEnv(task_name=task_name)
    obs = env.reset()

    print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}")
    sys.stdout.flush()

    rewards  = []
    history  = []
    step     = 0
    done     = False

    while not done:
        obs_dict = obs.model_dump()
        # Strip hidden ground-truth before sending to model
        for email in obs_dict["emails"]:
            email.pop("category", None)
            email.pop("priority",  None)

        raw_text = ""
        try:
            action_dict, raw_text = get_action(obs_dict, history)
            action = Action(**action_dict)
        except Exception as exc:
            err = f"parse_error:{exc}"
            visible = obs_dict["emails"]
            if not visible:
                break
            action = Action(email_id=visible[0]["id"], action="archive", value=None)
            # Log bad step and continue
            step += 1
            rewards.append(-0.05)
            print(f"[STEP] step={step} action=fallback_archive reward=-0.05 done=false error={err}")
            sys.stdout.flush()
            continue

        obs, reward, done, info = env.step(action)
        step += 1
        rewards.append(reward)

        last_error = info.get("last_action_error") or "null"
        done_str   = "true" if done else "false"
        action_str = f"{action.action}('{action.email_id}','{action.value}')"

        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={last_error}")
        sys.stdout.flush()

        history.append({"role": "user",      "content": json.dumps(obs_dict)})
        history.append({"role": "assistant", "content": raw_text})

    final_score  = env.final_score()
    success      = final_score >= 0.6
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={'true' if success else 'false'} steps={step} rewards={rewards_str}")
    sys.stdout.flush()


if __name__ == "__main__":
    tasks    = ["easy_classify", "medium_prioritize", "hard_triage"]
    task_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if task_arg and task_arg in tasks:
        run_task(task_arg)
    else:
        for task in tasks:
            run_task(task)
