from langgraph.graph import END, START, StateGraph
from .models import AgentState, Response
from .actors import executor_model, planner_model, replanner_model
from .tools import download_file_tool
from .util import save_graph

# create nodes
# Plan step
def plan_step(state: AgentState):
    plan = planner_model.invoke({"messages": [("user", state["question"])]})
    return {"plan": plan.steps, "has_file": plan.has_file}

# Download file
def download_file(state: AgentState):
  file_path = download_file_tool.invoke({"task_id": state["task_id"]})
  return {"attachment": file_path}

# Execute step
def execute_step(state: AgentState) -> AgentState:
  plan = state["plan"]
  plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
  task = plan[0]
  
  # Base prompt
  prompt_task_formatted = f"""For the following plan:\n{plan_str}\n\nYou are tasked with executing step {1}, {task}. (task_id: {state["task_id"]})"""
  
  # Add filepath if file exists
  prompt_task_formatted += f"\n\nFile available at: {state['attachment']}"
  
  # "create_react_agent" works with a messages state by default
  response = executor_model.invoke({"messages": [("user", prompt_task_formatted)]})
  return {"past_steps": [(task, response['messages'][-1].content)]}

# Replan step
def replan_step(state: AgentState):
  output = replanner_model.invoke(state)
  if isinstance(output.action, Response):
      return {"answer": output.action.response}
  else:
      return {"plan": output.action.steps}

def should_download(state: AgentState):
  if state["has_file"]:
      return "download_file"
  else:
      return "react_agent"

def should_end(state: AgentState):
  if "answer" in state and state["answer"]:
      return END
  else:
      return "react_agent"


def build_graph() -> StateGraph:
  # instantiate graph builder with state
  workflow = StateGraph(AgentState)

  # add nodes and edges
  workflow.add_node('planner', plan_step)
  workflow.add_node('download_file', download_file)
  workflow.add_conditional_edges(
     'planner',
     should_download,
     ['react_agent', 'download_file'])
  workflow.add_edge('download_file', 'react_agent')
  workflow.add_node('react_agent', execute_step)
  workflow.add_node('replanner', replan_step)
  workflow.add_edge(START, 'planner')
  workflow.add_edge('react_agent', 'replanner')
  workflow.add_conditional_edges(
    "replanner",
    should_end,
    ["react_agent", END],
)

  # compile graph. generate png image. store in current directory
  graph = workflow.compile()
  save_graph(graph, 'graph.png')

  return graph