import asyncio
import os
from typing import List, Optional, Dict
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from client import LegalRiskEnvClient
try:
    from models import LegalAction, LegalObservation
except ImportError:
    from .models import LegalAction, LegalObservation


API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "legal-risk-analysis")
OPENENV_URL = os.getenv("OPENENV_URL", "http://localhost:8000")
BENCHMARK = "legal-risk-analyzer"
MAX_STEPS = 8
TEMPERATURE = 0.4
HISTORY_WINDOW = 5  # last N steps passed to the LLM
MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99


#  Structured STDOUT loggers 
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action_repr: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    clean = action_repr.replace("'", "").replace('"', "")
    print(
        f"[STEP] step={step} action={clean} reward={reward:.2f} done={done_val} "
        f"error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _bounded_score(value: float) -> float:
    return max(MIN_TASK_SCORE, min(MAX_TASK_SCORE, value))


def _get_image_name() -> str:
    return (os.environ.get("LOCAL_IMAGE_NAME") or os.environ.get("IMAGE_NAME") or "").strip()


#  Action history helpers 
def _build_history_summary(action_history: List[Dict], window: int = HISTORY_WINDOW) -> str:
    if not action_history:
        return "  (none)"
    lines = []
    for i, h in enumerate(action_history[-window:], 1):
        lines.append(
            f"  Step {i}: action_type={h['action_type']} "
            f"text=\"{h['text'][:40]}...\" reward={h['reward']:.2f}"
        )
    return "\n".join(lines)


def _detect_greedy_loop(action_history: List[Dict], window: int = 3) -> Optional[str]:
    """Return the looping action_type if the last `window` actions are identical, else None."""
    if len(action_history) < window:
        return None
    recent = [h["action_type"] for h in action_history[-window:]]
    if len(set(recent)) == 1:
        return recent[0]
    return None


def _force_different_action(looping_type: str) -> str:
    """Cycle to a different action type to break the greedy loop."""
    cycle = ["extract", "classify", "rewrite"]
    idx = cycle.index(looping_type) if looping_type in cycle else 0
    return cycle[(idx + 1) % len(cycle)]


def _extract_candidate_clause(contract_text: str) -> str:
    lines = [ln.strip() for ln in contract_text.splitlines() if ln.strip()]
    if not lines:
        return contract_text.strip()

    priorities = [
        "limitation of liability",
        "indemnification",
        "terminate",
        "termination",
        "confidentiality",
    ]

    for keyword in priorities:
        for line in lines:
            if keyword in line.lower():
                return line

    # Fallback: choose the longest substantive line.
    return max(lines, key=len)


def _last_action_text(action_history: List[Dict], action_type: str) -> str:
    for item in reversed(action_history):
        if item["action_type"] == action_type:
            return item["text"]
    return ""


def _fallback_classify_clause_text(clause_text: str) -> str:
    text = clause_text.lower()
    if "limitation of liability" in text or "in no event shall" in text:
        return "high"
    if "indemn" in text:
        return "medium"
    if "termination for convenience" in text or "either party may terminate" in text:
        return "low"
    return "medium"


def _fallback_build_fair_rewrite(clause_text: str) -> str:
    text = clause_text.lower()
    if "indemn" in text:
        return (
            "Both parties agree that each party shall indemnify the other against third-party "
            "claims arising from its own negligence, misconduct, or breach of this Agreement."
        )
    if "terminate" in text:
        return (
            "Both parties agree that either party may terminate this Agreement for convenience upon "
            "60 days written notice, with mutual cooperation on an orderly transition."
        )
    return (
        "Neither party shall be liable to the other for any indirect, incidental, special, "
        "exemplary, or consequential damages, and both parties agree that any limitation of "
        "liability applies mutually and symmetrically."
    )


def _call_model(openai_client: OpenAI, system_prompt: str, user_prompt: str, max_tokens: int = 220) -> str:
    completion = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        stream=False,
    )
    return (completion.choices[0].message.content or "").strip()


def get_agent_action(
    openai_client: OpenAI,
    obs: LegalObservation,
    action_history: List[Dict],
) -> LegalAction:
    history_summary = _build_history_summary(action_history)
    if obs.task_id == 1:
        fallback = _extract_candidate_clause(obs.contract_text)
        try:
            reply = _call_model(
                openai_client,
                system_prompt=(
                    "You extract the exact target clause from a contract. "
                    "Reply with only the single clause text, verbatim, no quotes."
                ),
                user_prompt=(
                    f"Contract text:\n{obs.contract_text}\n\n"
                    f"Environment message: {obs.message}\n"
                    f"Action history:\n{history_summary}\n\n"
                    "Return the exact clause that appears to carry the main legal risk."
                ),
                max_tokens=160,
            )
            text = reply if reply else fallback
        except Exception as e:
            print(f"[DEBUG] Model extract failed: {e}", flush=True)
            text = fallback
        return LegalAction(action_type="extract", text_content=text)

    if obs.task_id == 2:
        extracted_clause = _last_action_text(action_history, "extract") or _extract_candidate_clause(obs.contract_text)
        fallback = _fallback_classify_clause_text(extracted_clause)
        try:
            reply = _call_model(
                openai_client,
                system_prompt=(
                    "You classify legal risk. Reply with exactly one lowercase token: "
                    "low, medium, or high."
                ),
                user_prompt=(
                    f"Clause:\n{extracted_clause}\n\n"
                    f"Environment message: {obs.message}\n"
                    f"Action history:\n{history_summary}\n\n"
                    "Choose the best risk label."
                ),
                max_tokens=8,
            ).lower()
            text = reply if reply in {"low", "medium", "high"} else fallback
        except Exception as e:
            print(f"[DEBUG] Model classify failed: {e}", flush=True)
            text = fallback
        return LegalAction(action_type="classify", text_content=text)

    clause_to_rewrite = _last_action_text(action_history, "extract") or _extract_candidate_clause(obs.contract_text)
    fallback = _fallback_build_fair_rewrite(clause_to_rewrite)
    try:
        reply = _call_model(
            openai_client,
            system_prompt=(
                "You rewrite legal clauses to be fair and mutual. "
                "Reply with only the rewritten clause text."
            ),
            user_prompt=(
                f"Original clause:\n{clause_to_rewrite}\n\n"
                f"Environment message: {obs.message}\n"
                f"Action history:\n{history_summary}\n\n"
                "Rewrite it so it is balanced for both parties. Include explicit mutual language "
                "such as 'both parties agree', 'each party', or 'neither party'."
            ),
            max_tokens=180,
        )
        text = reply if reply else fallback
    except Exception as e:
        print(f"[DEBUG] Model rewrite failed: {e}", flush=True)
        text = fallback
    return LegalAction(action_type="rewrite", text_content=text)


def _sanitize_action(action: LegalAction) -> str:
    clean_text = " ".join(action.text_content.split())
    return f"{action.action_type}({clean_text[:80]})"


#  Main loop 
async def main() -> None:
    rewards: List[float] = []
    action_history: List[Dict] = []
    steps_taken = 0
    score = MIN_TASK_SCORE
    success = False
    env = None
    result = None
    final_done = False

    try:
        openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[DEBUG] Failed to initialize OpenAI client: {e}", flush=True)
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, score=MIN_TASK_SCORE, rewards=[])
        return

    image_name = _get_image_name()
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        print(f"[DEBUG] Connecting to {OPENENV_URL}", flush=True)
        env = LegalRiskEnvClient(base_url=OPENENV_URL)
        result = await env.reset()
    except Exception as http_error:
        if not image_name:
            raise RuntimeError(f"Failed to connect to OpenEnv server at {OPENENV_URL}: {http_error}") from http_error
        print(f"[DEBUG] HTTP connection failed, trying Docker image: {image_name}", flush=True)
        env = await LegalRiskEnvClient.from_docker_image(image_name)

    try:
        if result is None:
            result = await env.reset()
        obs = result.observation
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            action = get_agent_action(openai_client, obs, action_history)
            result = await env.step(action)
            obs = result.observation
            reward = _bounded_score(result.reward or MIN_TASK_SCORE)
            done = result.done
            error = None
            # Record history for loop-breaker
            action_history.append({
                "action_type": action.action_type,
                "text": action.text_content,
                "reward": reward,
            })
            rewards.append(reward)
            steps_taken = step
            final_done = done
            log_step(
                step=step,
                action_repr=_sanitize_action(action),
                reward=reward,
                done=done,
                error=error,
            )
            if done:
                break
        score = _bounded_score(sum(rewards) / 3.0) if rewards else MIN_TASK_SCORE
        success = final_done and score >= 0.5
    except Exception as e:
        print(f"[DEBUG] Inference loop failed: {e}", flush=True)
    finally:
        try:
            if env is not None:
                await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
if __name__ == "__main__":
    asyncio.run(main())
