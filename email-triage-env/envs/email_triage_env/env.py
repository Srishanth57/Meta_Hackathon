"""
Email Triage Environment — OpenEnv Implementation
Real-world task: classify, route, and respond to emails.
"""

import random
from typing import Optional
from pydantic import BaseModel, Field


# ── Typed Models ──────────────────────────────────────────────────────────────

class EmailObservation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    task: str  # which sub-task is active: classify-email | route-email | full-triage
    step: int = 0


class EmailAction(BaseModel):
    urgency: str = Field(..., description="low | medium | high | critical")
    category: str = Field(..., description="billing | technical | general | complaint | spam")
    department: Optional[str] = Field(None, description="support | finance | engineering | hr | management")
    reply: Optional[str] = Field(None, description="Draft reply text (required for full-triage)")


class EmailReward(BaseModel):
    value: float
    breakdown: dict


# ── Email Dataset ─────────────────────────────────────────────────────────────

EMAILS = [
    {
        "email_id": "E001",
        "subject": "URGENT: Payment failed, service suspended",
        "body": (
            "Hi, my payment failed this morning and my entire team has been locked out "
            "of the platform. We have a critical demo in 2 hours. Please restore access NOW. "
            "Order #98231. This is unacceptable."
        ),
        "sender": "cto@acmecorp.com",
        "timestamp": "2025-04-08T08:14:00Z",
        "ground_truth": {
            "urgency": "critical",
            "category": "billing",
            "department": "finance",
            "reply_keywords": ["apologize", "payment", "restore", "team", "priority"],
        },
    },
    {
        "email_id": "E002",
        "subject": "Question about invoice format",
        "body": (
            "Hello, I was wondering if it's possible to receive invoices in PDF format "
            "instead of HTML. No rush, just a preference for our accounting system."
        ),
        "sender": "accounting@smallbiz.io",
        "timestamp": "2025-04-08T09:30:00Z",
        "ground_truth": {
            "urgency": "low",
            "category": "billing",
            "department": "finance",
            "reply_keywords": ["pdf", "invoice", "accounting", "format"],
        },
    },
    {
        "email_id": "E003",
        "subject": "API returning 500 errors since last deploy",
        "body": (
            "Your API has been returning HTTP 500 on /v2/users/sync since yesterday's "
            "deployment. Our integration is completely broken. Error logs attached. "
            "Stack trace: NullPointerException at UserSyncService:142."
        ),
        "sender": "dev@partner.net",
        "timestamp": "2025-04-08T10:05:00Z",
        "ground_truth": {
            "urgency": "high",
            "category": "technical",
            "department": "engineering",
            "reply_keywords": ["500", "api", "bug", "investigate", "engineer"],
        },
    },
    {
        "email_id": "E004",
        "subject": "Win a free iPhone!!!",
        "body": (
            "Congratulations! You've been selected to win a free iPhone 15 Pro. "
            "Click here to claim your prize: http://totally-not-phishing.ru/win"
        ),
        "sender": "noreply@fakeprizes.ru",
        "timestamp": "2025-04-08T11:00:00Z",
        "ground_truth": {
            "urgency": "low",
            "category": "spam",
            "department": "support",
            "reply_keywords": [],
        },
    },
    {
        "email_id": "E005",
        "subject": "Feedback on your customer support experience",
        "body": (
            "I've been using your product for six months and I'm quite disappointed "
            "with the support response times. Last week I waited 3 days for a reply "
            "to a simple question. I'm considering switching providers."
        ),
        "sender": "user123@gmail.com",
        "timestamp": "2025-04-08T12:20:00Z",
        "ground_truth": {
            "urgency": "medium",
            "category": "complaint",
            "department": "support",
            "reply_keywords": ["apologize", "support", "response time", "improve", "value"],
        },
    },
]

TASK_IDS = ["classify-email", "route-email", "full-triage"]

URGENCY_SCORES = {
    "critical": {"critical": 1.0, "high": 0.5, "medium": 0.2, "low": 0.0},
    "high":     {"critical": 0.5, "high": 1.0, "medium": 0.5, "low": 0.0},
    "medium":   {"critical": 0.2, "high": 0.5, "medium": 1.0, "low": 0.5},
    "low":      {"critical": 0.0, "high": 0.0, "medium": 0.5, "low": 1.0},
}


# ── Environment ───────────────────────────────────────────────────────────────

class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage Environment.

    Tasks
    -----
    classify-email  (easy)   — predict urgency + category
    route-email     (medium) — predict urgency + category + department
    full-triage     (hard)   — predict all three + draft a reply
    """

    MAX_STEPS = 3  # agent gets up to 3 attempts per episode

    def __init__(self, task: str = "full-triage", email_id: Optional[str] = None):
        if task not in TASK_IDS:
            raise ValueError(f"task must be one of {TASK_IDS}")
        self.task = task
        self._email_id = email_id
        self._email: dict = {}
        self._step = 0
        self._done = False
        self._last_reward = 0.0
        self._rewards: list[float] = []
        self._last_action_error: Optional[str] = None

    # ── OpenEnv interface ──────────────────────────────────────────────────────

    def reset(self) -> EmailObservation:
        self._step = 0
        self._done = False
        self._rewards = []
        self._last_reward = 0.0
        self._last_action_error = None

        if self._email_id:
            matches = [e for e in EMAILS if e["email_id"] == self._email_id]
            self._email = matches[0] if matches else random.choice(EMAILS)
        else:
            self._email = random.choice(EMAILS)

        return self._make_obs()

    def step(self, action: EmailAction) -> tuple[EmailObservation, float, bool, dict]:
        if self._done:
            self._last_action_error = "Episode already finished. Call reset()."
            return self._make_obs(), 0.0, True, self._info()

        self._step += 1
        reward, breakdown = self._grade(action)
        self._last_reward = reward
        self._rewards.append(reward)
        self._last_action_error = None

        # Done when perfect score OR max steps reached
        if reward >= 1.0 or self._step >= self.MAX_STEPS:
            self._done = True

        obs = self._make_obs()
        return obs, reward, self._done, {**self._info(), "breakdown": breakdown}

    def state(self) -> dict:
        return {
            "email": self._email,
            "task": self.task,
            "step": self._step,
            "done": self._done,
            "rewards": self._rewards,
        }

    def close(self):
        self._done = True

    # ── Internal ───────────────────────────────────────────────────────────────

    def _make_obs(self) -> EmailObservation:
        return EmailObservation(
            email_id=self._email.get("email_id", ""),
            subject=self._email.get("subject", ""),
            body=self._email.get("body", ""),
            sender=self._email.get("sender", ""),
            timestamp=self._email.get("timestamp", ""),
            task=self.task,
            step=self._step,
        )

    def _grade(self, action: EmailAction) -> tuple[float, dict]:
        gt = self._email["ground_truth"]
        breakdown = {}

        # ── Urgency score (partial credit) ─────────────────────────────────────
        urgency_score = URGENCY_SCORES.get(gt["urgency"], {}).get(action.urgency, 0.0)
        breakdown["urgency"] = urgency_score

        # ── Category score ────────────────────────────────────────────────────
        category_score = 1.0 if action.category == gt["category"] else 0.0
        breakdown["category"] = category_score

        if self.task == "classify-email":
            total = (urgency_score * 0.5) + (category_score * 0.5)
            return round(total, 2), breakdown

        # ── Department score ──────────────────────────────────────────────────
        dept_score = 1.0 if action.department == gt["department"] else 0.0
        breakdown["department"] = dept_score

        if self.task == "route-email":
            total = (urgency_score * 0.35) + (category_score * 0.35) + (dept_score * 0.30)
            return round(total, 2), breakdown

        # ── Reply score (full-triage) ─────────────────────────────────────────
        reply_score = self._score_reply(action.reply or "", gt["reply_keywords"])
        breakdown["reply"] = reply_score

        # Penalize empty replies — but only for non-spam (spam should NOT reply)
        if gt["reply_keywords"] and (not action.reply or len(action.reply.strip()) < 20):
            reply_score = 0.0
            breakdown["reply"] = 0.0
            self._last_action_error = "Reply is too short or empty."

        total = (
            (urgency_score * 0.25)
            + (category_score * 0.25)
            + (dept_score * 0.20)
            + (reply_score * 0.30)
        )
        return round(total, 2), breakdown

    @staticmethod
    def _score_reply(reply: str, keywords: list[str]) -> float:
        if not keywords:
            # Spam — no reply needed; perfect score for not replying
            return 1.0 if (reply is None or len(reply.strip()) < 10) else 0.3
        reply_lower = reply.lower()
        hits = sum(1 for kw in keywords if kw.lower() in reply_lower)
        return round(hits / len(keywords), 2)

    def _info(self) -> dict:
        return {
            "task": self.task,
            "step": self._step,
            "done": self._done,
            "last_action_error": self._last_action_error,
        }
