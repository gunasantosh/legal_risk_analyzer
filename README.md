---
title: Legal Risk Analyzer
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: true
license: mit
---

# Legal Risk Analyzer

Legal Risk Analyzer is an OpenEnv-compatible reinforcement learning environment for legal contract analysis.  
It is designed for:

1. Automated evaluation in OpenEnv/benchmark pipelines.
2. Future policy optimization workflows (for example, GRPO-style training).
3. Human demo workflows via the mounted Gradio interface.

The environment runs a 3-step legal reasoning loop over sampled contract scenarios:

1. Clause extraction
2. Risk classification
3. Clause mitigation rewrite

---

## What This Environment Does

Each episode samples one benchmark scenario from a curated in-code dataset (`BENCHMARK_DATA`) that combines:

1. CUAD-style extraction targets (`target_clause`)
2. LexGLUE-style classification labels (`legal_label`)

Every scenario includes:

1. `contract_text`
2. `target_clause`
3. `legal_label`

This replaces the previous static single-contract setup and provides diversity for RL training/evaluation.

---

## Observation and Action Schema

### Action (`LegalAction`)

- `action_type`: literal enum of `extract | classify | rewrite`
- `text_content`: model output/action payload

Action validation is strict (`strict=True`, `extra="forbid"`).

### Observation (`LegalObservation`)

Each response includes:

1. `contract_text` (current sampled scenario text)
2. `task_id` (1, 2, or 3)
3. `reward` (scalar float from the latest step)
4. `done` (episode terminal flag)
5. `current_risk_assessment` (set after successful classification)
6. `message` (human-readable task feedback)
7. `metadata` (includes step count)

---

## Reward Logic

### Task 1: Extraction (`task_id=1`)

- Expected action: `extract`
- Reward: Token F1 between `action.text_content` and `current_scenario["target_clause"]`
- Transition rule: if `F1 > 0.8`, advance to Task 2

### Task 2: Classification (`task_id=2`)

- Expected action: `classify`
- Reward: `0.99` if normalized prediction exactly matches `current_scenario["legal_label"]`, else `0.01`
- Transition rule: on match, set `current_risk_assessment` and advance to Task 3

### Task 3: Mitigation Rewrite (`task_id=3`)

- Expected action: `rewrite`
- Reward engine:
  - Primary: keyword heuristic (mutual/fair phrasing)
  - Secondary: LLM fallback scorer for borderline cases
- Terminal rule (`done=True`): only when `reward > 0.7`

This guarantees scalar rewards and a clean terminal condition for RL.

---

## Reset/Step Behavior

### `POST /reset`

- Creates a new episode id
- Resets internal state
- Randomly samples a scenario from `BENCHMARK_DATA`
- Returns observation with:
  - `message`: `Environment reset successful. Episode: <id>`
  - `reward`: `0.01`
  - `task_id`: `1`

### `POST /step`

- Increments step count
- Applies task-specific reward logic
- Returns updated observation with explicit `message` + `reward`
- Invalid action-type for current task returns `+0.01` reward with guidance message

---

## API and Runtime Layout

Port `8000` serves both machine and human interfaces:

1. OpenEnv API (root):
   - `POST /reset`
   - `POST /step`
   - `GET /state`
   - `GET /schema`
   - `WS /ws`
2. Gradio demo:
   - `GET /web`
3. Health/docs:
   - `GET /health`
   - `GET /docs`

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
```

Set at least:

```bash
HF_TOKEN=hf_your_token_here
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
API_BASE_URL=https://router.huggingface.co/v1
IMAGE_NAME=
OPENENV_URL=http://localhost:8000
```

### 2. Build and run

```bash
docker build -t legal_risk_analyzer_env:latest .
docker run -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  legal_risk_analyzer_env:latest
```

### 3. Validate endpoints

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{}"
curl -I http://localhost:8000/web
```

### 4. Run inference agent

```bash
uv sync
uv run python inference.py
```

Example log format:

```text
[START] task=legal-risk-analysis env=legal-risk-analyzer model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action='extract(...)' reward=0.93 done=false error=null message="Task 1 Complete: +0.93 Reward. Proceed to classification."
[STEP] step=2 action='classify(...)' reward=0.99 done=false error=null message="Task 2 Complete: +0.99 Reward. Proceed to mitigation rewrite."
[STEP] step=3 action='rewrite(...)' reward=0.75 done=true error=null message="Task 3 Complete: +0.75 Reward. Episode finished."
[END] success=true steps=3 score=0.890 rewards=0.93,0.99,0.75
```

---

## Project Structure

```text
legal_risk_analyzer/
  models.py               # LegalAction / LegalObservation / LegalState
  client.py               # OpenEnv client parser (observation-first reward/done/message)
  inference.py            # Agent loop that consumes task_id + message + reward
  gradio_app.py           # Human-facing demo UI
  server/
    app.py                # FastAPI + OpenEnv server + optional /web mount
    environment.py        # Core RL environment + benchmark sampler + reward engine
  openenv.yaml            # OpenEnv configuration
  pyproject.toml          # Dependencies and scripts
  Dockerfile              # Container build for Spaces/runtime
```

---

## Notes for RL Training

1. Reward is always scalar and included in observation JSON (`observation.reward`).
2. Task progression is explicit via `task_id`.
3. `contract_text` is visible on every step.
4. Terminal condition is deterministic (`Task 3 reward > 0.7`).
5. `message` provides dense textual feedback for debugging, logging, and policy shaping.
