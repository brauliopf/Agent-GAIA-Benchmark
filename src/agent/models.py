from typing_extensions import TypedDict, Annotated
from operator import add
from typing import List, Tuple, Union
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    """Agent state for LangGraph"""
    task_id: Annotated[str, 'The task_id of the question']
    question: Annotated[str, 'The question to answer']
    has_file: Annotated[bool, 'Whether the question has a file to download']
    attachment: Annotated[str, 'The attachment to the question']
    plan: Annotated[List[str], 'The plan to answer the question']
    past_steps: Annotated[List[Tuple], add, 'The past steps taken to answer the question']
    answer: Annotated[str, 'The answer to the question']

# Pydantic models for LangGraph (output interface for specific nodes)
class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
    has_file: bool = Field(
        description="whether the plan has a file to download"
    )

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

class FinalAnswer(BaseModel):
    """Final answer to the question."""
    answer: str = Field(
        description="The final answer to the question"
    )
