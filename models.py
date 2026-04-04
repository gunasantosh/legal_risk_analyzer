from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class LegalAction(Action):
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
