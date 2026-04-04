import os
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openai import Client

try:
    from models import LegalAction, LegalObservation
except ImportError:
    from .models import LegalAction, LegalObservation

CONTRACT_TEXT = """
SERVICES AGREEMENT

1. Services. Provider agrees to provide services.
2. Limitation of Liability. IN NO EVENT SHALL PROVIDER BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT.
3. Termination. This agreement may be terminated by either party with 30 days notice.
"""

GOLDEN_CLAUSE = "IN NO EVENT SHALL PROVIDER BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT."

# Keywords that indicate a fair, mutual rewrite
FAIR_REWRITE_KEYWORDS = [
    "neither party",
    "both parties",
    "mutual",
    "each party",
    "either party shall not",
    "no party",
    "symmetr",
]


def calculate_token_f1(pred: str, target: str) -> float:
    pred_tokens = set(pred.lower().split())
    target_tokens = set(target.lower().split())
    if not pred_tokens or not target_tokens:
        return 0.0
    common = pred_tokens.intersection(target_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    return 2 * (precision * recall) / (precision + recall)


class LegalRiskEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = 1
        self._current_risk = ""
        self._done = False

        api_base_url = os.environ.get("API_BASE_URL")
        hf_token = os.environ.get("HF_TOKEN") or "dummy"

        self.client = Client(
            base_url=api_base_url if api_base_url else None,
            api_key=hf_token,
            timeout=30.0
        )

    def reset(self) -> LegalObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = 1
        self._current_risk = ""
        self._done = False
        return self._build_obs(0.0)

    def step(self, action: LegalAction) -> LegalObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = 0.0

        if self._done:
            return self._build_obs(0.0)

        if self._task_id == 1 and action.action_type == "extract":
            f1 = calculate_token_f1(action.text_content, GOLDEN_CLAUSE)
            reward = f1
            if f1 > 0.8:
                self._task_id = 2

        elif self._task_id == 2 and action.action_type == "classify":
            pred = action.text_content.lower()
            if "high" in pred:
                reward = 1.0
                self._current_risk = "High"
                self._task_id = 3
            else:
                reward = 0.0

        elif self._task_id == 3 and action.action_type == "rewrite":
            reward = self._grade_rewrite(action.text_content)
            if reward > 0.5:
                self._done = True

        return self._build_obs(reward)

    def _build_obs(self, reward: float) -> LegalObservation:
        return LegalObservation(
            contract_text=CONTRACT_TEXT,
            task_id=self._task_id,
            current_risk_assessment=self._current_risk,
            done=self._done,
            reward=reward,
            metadata={"step": self._state.step_count}
        )

    def _grade_rewrite(self, text: str) -> float:
        """
        Grade a rewritten liability clause.

        Primary: keyword heuristic - deterministic and fast.
        Secondary: LLM scoring for borderline cases (1 keyword hit only).
        """
        #  Primary: keyword-based heuristic 
        text_lower = text.lower()
        hits = sum(1 for kw in FAIR_REWRITE_KEYWORDS if kw in text_lower)
        keyword_score = 0.75 if hits >= 2 else (0.4 if hits == 1 else 0.1)

        # 2 keywords  clearly mutual rewrite, accept immediately.
        if keyword_score >= 0.6:
            return keyword_score

        #  Secondary: LLM scoring for borderline cases 
        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        prompt = (
            "You are a legal fairness evaluator. Grade the following rewritten "
            "limitation-of-liability clause on a scale from 0.0 (completely one-sided) "
            "to 1.0 (fully mutual and fair to both parties).\n"
            "Output ONLY a single float number, nothing else.\n\n"
            f"Clause to grade:\n{text}"
        )

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=30.0
            )
            raw = (response.choices[0].message.content or "").strip()
            val = float(raw)
            llm_score = max(0.0, min(1.0, val))
            # Take the higher of heuristic and LLM scores for fairness.
            return max(keyword_score, llm_score)
        except Exception:
            return keyword_score


    @property
    def state(self) -> State:
        return self._state


if __name__ == "__main__":
    # Multi-mode deployment entry point (satisfies openenv validate check)
    try:
        from server.app import main as serve
    except ImportError:
        from app import main as serve  # type: ignore
    serve()
