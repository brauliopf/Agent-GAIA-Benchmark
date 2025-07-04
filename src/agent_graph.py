from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, NotRequired, Tuple, Union
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from operator import add
from tools import wikipedia_search_tool, tavily_search_tool
from util import save_graph
from langgraph.prebuilt import create_react_agent

import os

from pydantic import BaseModel, Field

load_dotenv()

# Define the agent state at module level
# The field can start as an empty list with Annotated
class AgentState(TypedDict):
  question: Annotated[str, 'The question to answer']
  attachments: Annotated[list[str], 'The attachments to the question']
  plan: Annotated[list[str], 'The plan to answer the question']
  past_steps: Annotated[list[Tuple], add, 'The past steps taken to answer the question']
  answer: Annotated[str, 'The answer to the question']

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: list[str] = Field(
        description="different steps to follow, should be in sorted order"
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


# Define LLM actors - Planner
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

planner_model = planner_prompt | ChatOpenAI( # The pipe operator chains the prompt to the next component (chat llm)
    model="gpt-4o", temperature=0.4
).with_structured_output(Plan)

# Define LLM actors - Executor
llm = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model="deepseek-r1-distill-llama-70b",
    temperature=0.6,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=3,
  )

tools = [wikipedia_search_tool, tavily_search_tool]
executor_prompt = "You are a helpful assistant."
executor_model = create_react_agent(llm, tools, prompt=executor_prompt)

# Define LLM actors - RePLan
replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{question}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)

def build_graph() -> StateGraph:
  # instantiate graph builder with state
  workflow = StateGraph(AgentState)
  
  # create nodes
  # Plan step
  def plan_step(state: AgentState):
      print(f"Enter Plan step: {state['question']}")
      plan = planner_model.invoke({"messages": [("user", state["question"])]})
      return {"plan": plan.steps}

  # Execute step
  def execute_step(state: AgentState) -> AgentState:
    print(f"Enter Execute step: {state['question']}")
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    # "create_react_agent" works with a messages state by default
    response = executor_model.invoke({"messages": [("user", task_formatted)]})
    return {"past_steps": [(task, response['messages'][-1].content)]}
  
  # Replan step
  def replan_step(state: AgentState):
    output = replanner.invoke(state)
    if isinstance(output.action, Response):
        return {"answer": output.action.response}
    else:
        return {"plan": output.action.steps}
    

  def create_final_answer(state: AgentState) -> AgentState:
    return {
      'answer': state['answer']
    }
  
  def should_end(state: AgentState):
    if "answer" in state and state["answer"]:
        return END
    else:
        return "execute"

  # add nodes and edges
  workflow.add_node('planner', plan_step)
  workflow.add_node('react_agent', execute_step)
  workflow.add_node('replanner', replan_step)
  workflow.add_edge(START, 'planner')
  workflow.add_edge('planner', 'react_agent')
  workflow.add_edge('react_agent', 'replanner')
  workflow.add_conditional_edges(
    "replanner",
    should_end,
    ["react_agent", END],
)

  # compile graph. generate png image. store in current directory
  agent_graph = workflow.compile()
  save_graph(agent_graph, 'agent_graph.png')

  return agent_graph