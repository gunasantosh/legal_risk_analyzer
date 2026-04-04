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



# [Scale] Legal Risk Analyzer



> **Meta  Hugging Face Hackathon** * Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework



An RL-powered environment that systematically parses, risk-scores, and rewrites predatory legal clauses - protecting users from unfair, one-sided Terms of Service agreements.



##  Architecture



```

Port 8000 (HF Space)

 /           FastAPI + OpenEnv API  (machine-facing: POST /reset, /step)

 /web        Gradio UI              (human-facing: judge demo interface)

 /health     Health check endpoint

 /docs       OpenAPI documentation

```



**Machine First**: The OpenEnv API runs at the root so automated graders can ping `/reset` directly.  

**Human Second**: Judges can visit `/web` for a visual 3-step walkthrough.



##  Agent Pipeline



```

Contract Text

    

    

[Step 1] EXTRACT   Token F1 vs golden clause  reward  [0, 1]

    

    

[Step 2] CLASSIFY  UNFAIR-ToS risk level  reward  {0, 1}

    

    

[Step 3] REWRITE   Keyword heuristic + LLM  reward  {0.1, 0.4, 0.75}

```



##  Quick Start



### Prerequisites



- Docker Desktop

- A Hugging Face account with API access

- `uv` package manager (`pip install uv`)



### 1. Configure Environment



```bash

cp .env.example .env

# Edit .env and fill in your HF_TOKEN

```



Required keys:



```bash

HF_TOKEN=hf_your_token_here

MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

API_BASE_URL=https://router.huggingface.co/v1

IMAGE_NAME=          # leave blank to use localhost:8000

```



### 2. Build & Run the Docker Container



```bash

docker build -t legal_risk_analyzer_env:latest .

docker run -p 8000:8000 \

  -e HF_TOKEN=$HF_TOKEN \

  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \

  -e API_BASE_URL=https://router.huggingface.co/v1 \

  legal_risk_analyzer_env:latest

```



### 3. Verify



```bash

# OpenEnv API health check

curl http://localhost:8000/health



# POST to reset (grader ping)

curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{}"



# Gradio UI - open in browser

open http://localhost:8000/web

```



### 4. Run the Agent (Inference)



```bash

uv sync

uv run python inference.py

```



Expected output:

```

[START] task=legal-risk-analysis env=legal-risk-analyzer model=Qwen/Qwen2.5-72B-Instruct

[STEP] step=1 action='extract(IN NO EVENT SHALL PR...)' reward=1.00 done=false error=null

[STEP] step=2 action='classify(High...)' reward=1.00 done=false error=null

[STEP] step=3 action='rewrite(Neither party shall...)' reward=0.75 done=true error=null

[END] success=true steps=3 score=0.917 rewards=1.00,1.00,0.75

```



##  Validation



```bash

# OpenEnv spec compliance

openenv validate



# Docker build

docker build -t legal_risk_analyzer_env:latest .



# API readiness

curl -X POST http://localhost:8000/reset



# UI readiness

curl -I http://localhost:8000/web

```



##  File Structure



```

legal_risk_analyzer/

 inference.py          # History-aware agent (loop-breaker logic)

 gradio_app.py         # Gradio judge UI (mounted at /web)

 client.py             # OpenEnv HTTP client wrapper

 server/

    app.py            # FastAPI + Gradio mount

    environment.py    # RL environment (keyword grader + LLM fallback)

    models.py         # Pydantic action/observation schemas

 Dockerfile            # Multi-stage build (openenv-base)

 pyproject.toml        # Dependencies (openenv-core, openai, gradio)

 openenv.yaml          # OpenEnv spec config

 .env                  # Local secrets (not committed)

```



##  Infrastructure Constraints



| Constraint | Value |

|---|---|

| Runtime | < 20 min for 10-step loop |

| RAM | < 8 GB (no local models) |

| Model | Qwen/Qwen2.5-72B-Instruct via HF Inference API |

| vCPUs | 2 (Docker optimized) |

