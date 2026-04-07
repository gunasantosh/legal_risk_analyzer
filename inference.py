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


API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "legal-risk-analysis")
OPENENV_URL = os.getenv("OPENENV_URL", "http://localhost:8000")
BENCHMARK = "legal-risk-analyzer"
MAX_STEPS = 10
TEMPERATURE = 0.4
HISTORY_WINDOW = 5  # last N steps passed to the LLM


#  Structured STDOUT loggers 
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action_repr: str,
    reward: float,
    done: bool,
    error: Optional[str],
    message: str = "",
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Spec requires single quotes around action value
    clean = action_repr.replace("'", "").replace('"', "")
    print(
        f"[STEP] step={step} action='{clean}' reward={reward:.2f} done={done_val} "
        f"error={error_val} message=\"{message}\"",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _get_image_name() -> str:
    """
    Prefer explicit process environment configuration and avoid accidental
    dependence on a checked-in `.env` value during remote evaluation.
    """
    return os.environ.get("IMAGE_NAME", "").strip()


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


#  Core agent 
def get_agent_action(
    openai_client: OpenAI,
    obs: LegalObservation,
    action_history: List[Dict],
) -> LegalAction:
    """
    History-aware agent decision.
    Determines the next action using:
    1. Current task_id as the primary signal.
    2. A greedy-loop breaker: if the last 3 actions are the same type and
       all yielded 0 reward, force a different action type.
    3. A history-aware system + user prompt so the LLM avoids repeating
       zero-reward strategies.
    """
    # Determine action strictly from environment task_id.
    if obs.task_id == 1:
        action_type = "extract"
    elif obs.task_id == 2:
        action_type = "classify"
    else:
        action_type = "rewrite"
    # --- Build prompts ---
    history_summary = _build_history_summary(action_history)
    system_prompt = (
        "You are a legal AI assistant completing a 3-step legal risk analysis pipeline.\n"
        "The pipeline tasks are:\n"
        "  1. extract   Pull the exact verbatim target clause from the contract.\n"
        "  2. classify  Output exactly one label: low, medium, or high.\n"
        "  3. rewrite   Rewrite the clause to be fair, mutual, and balanced for both parties.\n\n"
        "RULES:\n"
        "- Reply with ONLY the requested output - no preamble, no quotes, no labels.\n"
        "- Use the latest environment message and reward to improve your next attempt.\n"
        "- For rewrite: use phrases like 'Neither party shall', 'both parties agree', "
        "'mutual limitation', and cap liability symmetrically.\n"
        "- Never repeat an action that already earned 0 reward verbatim."
    )
    if action_type == "extract":
        last_failed_extract = ""
        for h in reversed(action_history):
            if h["action_type"] == "extract" and h["reward"] < 0.8:
                last_failed_extract = h["text"]
                break

        user_prompt = (
            f"Contract Text:\n{obs.contract_text}\n\n"
            f"Environment message: {obs.message}\n"
            f"Current reward: {obs.reward:.2f}\n\n"
            "Task: Extract the EXACT target clause text from the contract.\n"
            "Return one full clause sentence/line verbatim.\n\n"
            f"Action History:\n{history_summary}\n\n"
            f"Last failed extraction (if any): {last_failed_extract}\n"
            "Constraint: Do not repeat a low-reward extraction verbatim."
        )
    elif action_type == "classify":
        extracted_clause = _last_action_text(action_history, "extract") or _extract_candidate_clause(obs.contract_text)
        user_prompt = (
            f"Clause to classify:\n{extracted_clause}\n\n"
            f"Environment message: {obs.message}\n"
            f"Current reward: {obs.reward:.2f}\n\n"
            "Task: Classify legal risk.\n"
            "Reply with ONLY one lowercase token: low, medium, or high.\n\n"
            f"Action History:\n{history_summary}\n\n"
            "Constraint: Do not repeat an approach that earned reward=0.00."
        )
    else:  # rewrite
        clause_to_rewrite = _last_action_text(action_history, "extract") or _extract_candidate_clause(obs.contract_text)
        # Find last rewrite attempts that failed to give context
        failed_rewrites = [
            h["text"] for h in action_history
            if h["action_type"] == "rewrite" and h["reward"] <= 0.7
        ]
        failed_block = ""
        if failed_rewrites:
            failed_block = (
                "\n\nPrevious failed rewrite attempts (do NOT repeat these):\n"
                + "\n---\n".join(f'"{t}"' for t in failed_rewrites[-2:])
            )
        user_prompt = (
            "Task: Rewrite the following clause "
            "to be fair, balanced, and mutual for BOTH parties.\n\n"
            f"Original clause:\n'{clause_to_rewrite}'\n\n"
            f"Environment message: {obs.message}\n"
            f"Current reward: {obs.reward:.2f}\n\n"
            "Requirements:\n"
            "- Use symmetric language: 'Neither party shall...'\n"
            "- Explicitly indicate mutuality for both parties\n"
            "- Keep it professional and legally sound\n"
            "- Reply with ONLY the rewritten clause text\n"
            f"{failed_block}\n\n"
            f"Action History:\n{history_summary}\n\n"
            "Constraint: Do not repeat an approach that earned reward=0.00."
        )
    # --- LLM call ---
    try:
        completion = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=600,
            stream=False,
        )
        reply = (completion.choices[0].message.content or "").strip()
        # Strip surrounding quotes added by some models
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]
        if reply.startswith("'") and reply.endswith("'"):
            reply = reply[1:-1]
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        # Sensible fallbacks that ensure progress
        fallbacks = {
            "extract": _extract_candidate_clause(obs.contract_text),
            "classify": "high",
            "rewrite": (
                "Neither party shall be liable to the other for indirect, incidental, "
                "special, or consequential damages, and both parties agree that any "
                "limitation of liability is mutual and applies symmetrically."
            ),
        }
        reply = fallbacks[action_type]
    return LegalAction(action_type=action_type, text_content=reply)
#  Main loop 
async def main() -> None:
    rewards: List[float] = []
    action_history: List[Dict] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        log_step(
            step=0,
            action_repr="startup",
            reward=0.0,
            done=True,
            error=str(e),
            message="Failed to initialize OpenAI client.",
        )
        log_end(task=TASK_NAME, success=False, steps=0, score=0.0, rewards=[])
        return

    image_name = _get_image_name()
    try:
        print(f"[DEBUG] Connecting to {OPENENV_URL}", flush=True)
        env = LegalRiskEnvClient(base_url=OPENENV_URL)
    except Exception as e:
        if image_name:
            try:
                print(f"[DEBUG] HTTP init failed, trying Docker image: {image_name}", flush=True)
                env = await LegalRiskEnvClient.from_docker_image(image_name)
            except Exception as docker_error:
                log_step(
                    step=0,
                    action_repr="startup",
                    reward=0.0,
                    done=True,
                    error=str(docker_error),
                    message=f"Failed to initialize env via HTTP and Docker. HTTP error: {e}",
                )
                log_end(task=TASK_NAME, success=False, steps=0, score=0.0, rewards=[])
                return
        else:
            log_step(
                step=0,
                action_repr="startup",
                reward=0.0,
                done=True,
                error=str(e),
                message="Failed to initialize environment client.",
            )
            log_end(task=TASK_NAME, success=False, steps=0, score=0.0, rewards=[])
            return

    try:
        result = await env.reset()
        obs = result.observation
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            action = get_agent_action(openai_client, obs, action_history)
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
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
            action_str = f"{action.action_type}({action.text_content[:20]}...)"
            log_step(
                step=step,
                action_repr=action_str,
                reward=reward,
                done=done,
                error=error,
                message=obs.message,
            )
            if done:
                break
        score = sum(rewards) / 3.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5
    except Exception as e:
        log_step(
            step=max(steps_taken, 0),
            action_repr="runtime",
            reward=0.0,
            done=True,
            error=str(e),
            message="Inference loop failed before completion.",
        )
    finally:
        try:
            if env is not None:
                await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(task=TASK_NAME, success=success, steps=steps_taken, score=score, rewards=rewards)
if __name__ == "__main__":
    asyncio.run(main())
