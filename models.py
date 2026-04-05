from typing import Literal
from openenv.core.env_server.types import Action, Observation, State
from pydantic import ConfigDict, Field

class LegalState(State):
    episode_id: str = Field(..., description="Unique identifier for the episode")
    step_count: int = Field(0, description="Number of steps taken in the current episode")

class LegalAction(Action):
    model_config = ConfigDict(strict=True, extra="forbid")

    action_type: Literal["extract", "classify", "rewrite"] = Field(
        ..., description="Type of action to perform"
    )
    text_content: str = Field(
        ..., description="Action content or selected text"
    )

class LegalObservation(Observation):
    contract_text: str = Field(
        default="", description="The text of the contract for the current task"
    )
    task_id: int = Field(
        default=1, description="Current task ID (1, 2, or 3)"
    )
    current_risk_assessment: str = Field(
        default="", description="Risk assessment string if applicable"
    )
    done: bool = Field(
        default=False, description="Whether all tasks are completed"
    )
    reward: float = Field(
        default=0.0, description="Scalar reward from the last action"
    )
    message: str = Field(
        default="", description="Human-readable environment feedback message"
    )
