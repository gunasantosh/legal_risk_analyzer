import os
import random
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()
from openenv.core.env_server.interfaces import Environment
from openai import Client
try:
    from models import LegalAction, LegalObservation, LegalState
except ImportError:
    from .models import LegalAction, LegalObservation, LegalState  


BENCHMARK_DATA = [
    {
        "source": "CUAD + LexGLUE",
        "contract_text": (
            "MASTER SERVICES AGREEMENT\n"
            "1. Fees. Customer shall pay all undisputed invoices within 30 days.\n"
            "2. Limitation of Liability. In no event shall Provider be liable for any "
            "indirect, incidental, special, exemplary, or consequential damages.\n"
            "3. Governing Law. This Agreement is governed by the laws of New York."
        ),
        "target_clause": (
            "In no event shall Provider be liable for any indirect, incidental, special, "
            "exemplary, or consequential damages."
        ),
        "legal_label": "high",
    },
    {
        "source": "CUAD + LexGLUE",
        "contract_text": (
            "SOFTWARE LICENSE AGREEMENT\n"
            "1. License Grant. Licensor grants a non-exclusive license to use the Software.\n"
            "2. Data Processing. Parties shall comply with applicable data protection laws.\n"
            "3. Termination for Convenience. Either party may terminate this Agreement "
            "for convenience with 60 days written notice."
        ),
        "target_clause": (
            "Either party may terminate this Agreement for convenience with 60 days written notice."
        ),
        "legal_label": "low",
    },
    {
        "source": "CUAD + LexGLUE",
        "contract_text": (
            "SUPPLY AGREEMENT\n"
            "1. Delivery. Supplier will deliver goods pursuant to agreed schedules.\n"
            "2. Indemnification. Supplier shall indemnify Customer from third-party claims "
            "arising from Supplier's negligence.\n"
            "3. Confidentiality. Receiving party shall protect confidential information with "
            "reasonable safeguards."
        ),
        "target_clause": (
            "Supplier shall indemnify Customer from third-party claims arising from Supplier's negligence."
        ),
        "legal_label": "medium",
    },
]

# Keywords that indicate a fair, mutual rewrite
FAIR_REWRITE_KEYWORDS = [
    "neither party",
    "both parties",
    "mutual",
    "each party",
    "either party shall not",
    "no party",
    "symmetric",
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
        self._state = LegalState(episode_id=str(uuid4()), step_count=0)
        self._task_id = 1
        self._current_risk = ""
        self._done = False
        self.current_scenario = random.choice(BENCHMARK_DATA)
        api_base_url = os.environ.get("API_BASE_URL")
        hf_token = os.environ.get("HF_TOKEN") or "dummy"
        self.client = Client(
            base_url=api_base_url if api_base_url else None,
            api_key=hf_token,
            timeout=30.0
        )

    def reset(self) -> LegalObservation:
        self._state = LegalState(episode_id=str(uuid4()), step_count=0)
        self._task_id = 1
        self._current_risk = ""
        self._done = False
        self.current_scenario = random.choice(BENCHMARK_DATA)
        return self._build_obs(
            reward=0.0,
            message=f"Environment reset successful. Episode: {self._state.episode_id}",
        )
    
    def step(self, action: LegalAction) -> LegalObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = 0.0
        message = "No state change."

        if self._done:
            return self._build_obs(
                reward=0.0,
                message="Episode already complete. Call reset() to start a new episode.",
            )
        
        if self._task_id == 1 and action.action_type == "extract":
            f1 = calculate_token_f1(action.text_content, self.current_scenario["target_clause"])
            reward = f1
            if f1 > 0.8:
                self._task_id = 2
                message = f"Task 1 Complete: +{reward:.2f} Reward. Proceed to classification."
            else:
                message = f"Task 1 Attempt: +{reward:.2f} Reward. Improve extraction overlap."

        elif self._task_id == 2 and action.action_type == "classify":
            pred = " ".join(action.text_content.lower().split())
            expected = self.current_scenario["legal_label"].lower()
            if pred == expected:
                reward = 1.0
                self._current_risk = self.current_scenario["legal_label"].title()
                self._task_id = 3
                message = f"Task 2 Complete: +{reward:.2f} Reward. Proceed to mitigation rewrite."
            else:
                reward = 0.0
                message = (
                    f"Task 2 Attempt: +{reward:.2f} Reward. Predicted '{pred}', expected legal label."
                )

        elif self._task_id == 3 and action.action_type == "rewrite":
            reward = self._grade_rewrite(action.text_content)
            if reward > 0.7:
                self._done = True
                message = f"Task 3 Complete: +{reward:.2f} Reward. Episode finished."
            else:
                message = (
                    f"Task 3 Attempt: +{reward:.2f} Reward. Rewrite needs better mutuality/fairness."
                )
        else:
            message = (
                f"Invalid action '{action.action_type}' for task {self._task_id}: +0.00 Reward."
            )
                
        return self._build_obs(reward=reward, message=message)
    
    def _build_obs(self, reward: float, message: str) -> LegalObservation:
        return LegalObservation(
            contract_text=self.current_scenario["contract_text"],
            task_id=self._task_id,
            current_risk_assessment=self._current_risk,
            done=self._done,
            reward=reward,
            message=message,
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
    def state(self) -> LegalState:
        return self._state
    
    @property
    def name(self) -> str:
        return "Legal Risk Assessment Environment"
    

if __name__ == "__main__":
    # Multi-mode deployment entry point (satisfies openenv validate check)
    try:
        from server.app import main as serve
    except ImportError:
        from app import main as serve  # type: ignore
    serve()
