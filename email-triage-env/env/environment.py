"""
Email Triage Environment — OpenEnv compliant implementation.
Agents must classify, prioritize, and act on emails in an inbox.
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


# ── Pydantic models ──────────────────────────────────────────────────────────

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    category: Optional[str] = None          # ground-truth label (hidden from agent)
    priority: Optional[str] = None          # "high" | "medium" | "low"


class Observation(BaseModel):
    emails: List[Email]
    inbox_size: int
    step: int
    message: str


class Action(BaseModel):
    email_id: str
    action: str          # "classify" | "prioritize" | "reply" | "archive" | "escalate"
    value: Optional[str] = None   # label / priority level / reply text


class Reward(BaseModel):
    value: float
    reason: str


# ── Task definitions ─────────────────────────────────────────────────────────

TASKS = {
    "easy_classify": {
        "description": "Classify 5 emails into the correct category (spam/work/personal/newsletter).",
        "max_steps": 10,
        "emails": [
            Email(id="e1", sender="boss@company.com", subject="Q3 Report Due",
                  body="Please send the Q3 financial report by Friday EOD.", category="work", priority="high"),
            Email(id="e2", sender="promo@shop.com", subject="50% OFF everything!",
                  body="Huge sale this weekend only! Click here to claim your discount.", category="spam", priority="low"),
            Email(id="e3", sender="mom@gmail.com", subject="Sunday dinner",
                  body="Are you coming for dinner on Sunday? Grandma will be here!", category="personal", priority="medium"),
            Email(id="e4", sender="newsletter@techdigest.io", subject="This week in AI",
                  body="Top stories: LLMs in production, new benchmarks, and more.", category="newsletter", priority="low"),
            Email(id="e5", sender="hr@company.com", subject="Benefits enrollment deadline",
                  body="Benefits enrollment closes in 3 days. Log in to make selections.", category="work", priority="high"),
        ],
    },
    "medium_prioritize": {
        "description": "Read 6 emails and assign correct priority (high/medium/low) then archive low-priority ones.",
        "max_steps": 15,
        "emails": [
            Email(id="m1", sender="ceo@company.com", subject="Board meeting prep",
                  body="We need the slides for tomorrow's board meeting. Critical.", category="work", priority="high"),
            Email(id="m2", sender="friend@gmail.com", subject="Weekend plans",
                  body="Hey! Want to grab coffee this Saturday?", category="personal", priority="low"),
            Email(id="m3", sender="security@bank.com", subject="Suspicious login detected",
                  body="We noticed a login from an unrecognized device. Verify now.", category="work", priority="high"),
            Email(id="m4", sender="newsletter@medium.com", subject="Your weekly digest",
                  body="Here are this week's top stories curated for you.", category="newsletter", priority="low"),
            Email(id="m5", sender="client@bigcorp.com", subject="Contract renewal discussion",
                  body="We'd like to schedule a call to discuss the upcoming contract renewal.", category="work", priority="medium"),
            Email(id="m6", sender="promo@games.com", subject="You won a prize!",
                  body="Congratulations! Click the link to claim your gift card.", category="spam", priority="low"),
        ],
    },
    "hard_triage": {
        "description": "Full triage: classify, prioritize, reply to high-priority work emails, and archive junk. 8 emails.",
        "max_steps": 25,
        "emails": [
            Email(id="h1", sender="boss@company.com", subject="Urgent: server is down",
                  body="Production server crashed. We need you on this NOW.", category="work", priority="high"),
            Email(id="h2", sender="promo@casino.com", subject="Free spins inside",
                  body="You have 100 free spins waiting. Click to play!", category="spam", priority="low"),
            Email(id="h3", sender="support@saas.io", subject="Your trial expires tomorrow",
                  body="Your 14-day trial ends tomorrow. Upgrade to keep access.", category="work", priority="medium"),
            Email(id="h4", sender="colleague@company.com", subject="Can you review my PR?",
                  body="Hi, could you review pull request #42 when you get a chance?", category="work", priority="medium"),
            Email(id="h5", sender="newsletter@startup.com", subject="Funding news roundup",
                  body="This week's biggest funding rounds in tech.", category="newsletter", priority="low"),
            Email(id="h6", sender="cto@company.com", subject="Security audit findings",
                  body="The security audit found 3 critical vulnerabilities. Immediate action required.", category="work", priority="high"),
            Email(id="h7", sender="mom@gmail.com", subject="Your birthday party",
                  body="We're planning a surprise for dad! Can you keep Saturday free?", category="personal", priority="medium"),
            Email(id="h8", sender="billing@vendor.com", subject="Invoice overdue",
                  body="Invoice #1042 is 30 days overdue. Please remit payment.", category="work", priority="high"),
        ],
    },
}


# ── Environment ───────────────────────────────────────────────────────────────

class EmailTriageEnv:
    """OpenEnv-compliant Email Triage Environment."""

    VALID_CATEGORIES = {"spam", "work", "personal", "newsletter"}
    VALID_PRIORITIES = {"high", "medium", "low"}
    VALID_ACTIONS = {"classify", "prioritize", "reply", "archive", "escalate"}

    def __init__(self, task_name: str = "easy_classify"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS)}")
        self.task_name = task_name
        self.task_cfg = TASKS[task_name]
        self._step = 0
        self._done = False
        self._rewards: List[float] = []
        self._inbox: Dict[str, Email] = {}
        self._actions_taken: Dict[str, List[str]] = {}  # email_id → actions
        self._last_action_error: Optional[str] = None
        self.reset()

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self._step = 0
        self._done = False
        self._rewards = []
        self._last_action_error = None
        # deep-copy emails (hide ground-truth from observation)
        self._inbox = {
            e.id: e.model_copy() for e in self.task_cfg["emails"]
        }
        self._actions_taken = {e.id: [] for e in self.task_cfg["emails"]}
        self._classified: Dict[str, str] = {}
        self._prioritized: Dict[str, str] = {}
        self._replied: set = set()
        self._archived: set = set()
        return self._observe("Inbox loaded. Begin triaging emails.")

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if self._done:
            return self._observe("Episode already done."), 0.0, True, self._info()

        self._step += 1
        self._last_action_error = None
        reward = 0.0

        email = self._inbox.get(action.email_id)
        if email is None:
            self._last_action_error = f"Unknown email_id: {action.email_id}"
            reward = -0.05
        elif action.action == "classify":
            reward = self._handle_classify(email, action.value)
        elif action.action == "prioritize":
            reward = self._handle_prioritize(email, action.value)
        elif action.action == "reply":
            reward = self._handle_reply(email, action.value)
        elif action.action == "archive":
            reward = self._handle_archive(email)
        elif action.action == "escalate":
            reward = self._handle_escalate(email)
        else:
            self._last_action_error = f"Unknown action: {action.action}"
            reward = -0.05

        # Small step penalty to discourage unnecessary steps
        reward -= 0.01

        self._rewards.append(round(reward, 2))
        self._done = self._check_done()

        obs = self._observe(f"Step {self._step} completed.")
        return obs, round(reward, 2), self._done, self._info()

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.task_name,
            "step": self._step,
            "done": self._done,
            "classified": self._classified,
            "prioritized": self._prioritized,
            "replied": list(self._replied),
            "archived": list(self._archived),
            "cumulative_reward": round(sum(self._rewards), 2),
        }

    # ── Action handlers ──────────────────────────────────────────────────────

    def _handle_classify(self, email: Email, label: Optional[str]) -> float:
        if label not in self.VALID_CATEGORIES:
            self._last_action_error = f"Invalid category: {label}"
            return -0.05
        self._classified[email.id] = label
        correct = email.category == label
        return 0.3 if correct else -0.1

    def _handle_prioritize(self, email: Email, level: Optional[str]) -> float:
        if level not in self.VALID_PRIORITIES:
            self._last_action_error = f"Invalid priority: {level}"
            return -0.05
        self._prioritized[email.id] = level
        correct = email.priority == level
        return 0.2 if correct else -0.1

    def _handle_reply(self, email: Email, text: Optional[str]) -> float:
        if not text or len(text.strip()) < 10:
            self._last_action_error = "Reply text too short (< 10 chars)"
            return -0.05
        if email.id in self._replied:
            return -0.02  # already replied
        if email.category not in ("work", "personal"):
            return -0.1   # replying to spam/newsletter is bad
        self._replied.add(email.id)
        # Bonus for replying to high-priority work email
        if email.priority == "high" and email.category == "work":
            return 0.4
        return 0.2

    def _handle_archive(self, email: Email) -> float:
        if email.id in self._archived:
            return -0.02
        self._archived.add(email.id)
        # Correct to archive spam/newsletter; wrong to archive high-priority work
        if email.category in ("spam", "newsletter"):
            return 0.2
        if email.priority == "high" and email.category == "work":
            return -0.3   # penalty: archiving critical email
        return 0.05

    def _handle_escalate(self, email: Email) -> float:
        # Escalating high-priority work emails is good
        if email.priority == "high" and email.category == "work":
            return 0.3
        return -0.1  # unnecessary escalation

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _observe(self, message: str) -> Observation:
        visible_emails = [
            Email(id=e.id, sender=e.sender, subject=e.subject, body=e.body)
            for e in self._inbox.values()
            if e.id not in self._archived
        ]
        return Observation(
            emails=visible_emails,
            inbox_size=len(visible_emails),
            step=self._step,
            message=message,
        )

    def _check_done(self) -> bool:
        if self._step >= self.task_cfg["max_steps"]:
            return True
        all_ids = set(self._inbox.keys())
        # Done when every email has been classified + prioritized (or archived)
        for eid in all_ids:
            if eid not in self._archived:
                if eid not in self._classified or eid not in self._prioritized:
                    return False
        return True

    def _info(self) -> Dict[str, Any]:
        return {
            "last_action_error": self._last_action_error,
            "step": self._step,
            "cumulative_reward": round(sum(self._rewards), 2),
        }

    def final_score(self) -> float:
        """Normalised score in [0, 1] for grading."""
        emails = list(self._inbox.values())
        scores = []

        for e in emails:
            email_score = 0.0
            # Classification (0–0.4)
            if self._classified.get(e.id) == e.category:
                email_score += 0.4
            # Prioritization (0–0.3)
            if self._prioritized.get(e.id) == e.priority:
                email_score += 0.3
            # Action quality (0–0.3)
            if e.category == "spam" or e.category == "newsletter":
                if e.id in self._archived:
                    email_score += 0.3
            elif e.priority == "high" and e.category == "work":
                if e.id in self._replied or e.id in self._archived and False:
                    email_score += 0.3
                elif e.id in self._replied:
                    email_score += 0.3
            else:
                email_score += 0.15  # partial credit for non-critical emails
            scores.append(email_score)

        return round(sum(scores) / len(scores), 4) if scores else 0.0
