# Email Triage Environment — OpenEnv Hackathon Submission

An **OpenEnv-compliant** RL environment where an AI agent triages a realistic email inbox — classifying, prioritizing, replying to, archiving, and escalating emails.

## Motivation

Email overload is a universal productivity problem. This environment tests whether an LLM agent can autonomously triage an inbox the way a human knowledge worker would.

## Action Space

| Action | Value | Description |
|--------|-------|-------------|
| classify | spam/work/personal/newsletter | Label the email category |
| prioritize | high/medium/low | Set urgency level |
| reply | reply text | Send a reply |
| archive | null | Remove from inbox |
| escalate | null | Flag for immediate attention |

## Observation Space

Each step the agent sees: a list of emails (id, sender, subject, body), inbox_size, step number, and a message. Ground-truth labels are hidden.

## Tasks

| Task | Difficulty | Emails | Max Steps |
|------|-----------|--------|-----------|
| easy_classify | Easy | 5 | 10 |
| medium_prioritize | Medium | 6 | 15 |
| hard_triage | Hard | 8 | 25 |

## Reward Function

- Correct classification: +0.30, wrong: -0.10
- Correct priority: +0.20, wrong: -0.10
- Reply to high-priority work: +0.40
- Archive spam/newsletter: +0.20
- Archive high-priority work: -0.30 (penalty)
- Escalate critical work: +0.30
- Per-step penalty: -0.01

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
python inference.py
```

## Docker

```bash
docker build -t email-triage-env .
docker run -e HF_TOKEN=your_token email-triage-env
```

## Baseline Scores (gpt-4.1-mini)

| Task | Score | Success |
|------|-------|---------|
| easy_classify | 0.82 | YES |
| medium_prioritize | 0.74 | YES |
| hard_triage | 0.61 | YES |

## Output Format

```
[START] task=easy_classify env=email-triage model=gpt-4.1-mini
[STEP] step=1 action=classify('e1','work') reward=0.29 done=false error=null
[END] success=true steps=7 rewards=0.29,0.29,0.29,...
```

## Project Structure

```
email-triage-env/
├── inference.py      # Main submission script (root-level)
├── openenv.yaml      # OpenEnv metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── env/
    ├── __init__.py
    └── environment.py
```

## License: MIT
