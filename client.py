from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
try:
    from models import LegalAction, LegalObservation
except ImportError:
    from .models import LegalAction, LegalObservation


class LegalRiskEnvClient(EnvClient[LegalAction, LegalObservation, State]):
    def _step_payload(self, action: LegalAction) -> Dict:
        return {
            "action_type": action.action_type,
            "text_content": action.text_content,
        }
    
    def _parse_result(self, payload: Dict) -> StepResult[LegalObservation]:
        obs_data = payload.get("observation", {})
        observation = LegalObservation(
            contract_text=obs_data.get("contract_text", ""),
            task_id=obs_data.get("task_id", 1),
            current_risk_assessment=obs_data.get("current_risk_assessment", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )
    
    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
