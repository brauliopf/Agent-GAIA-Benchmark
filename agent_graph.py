from typing import Annotated, TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def build_graph():
  # define an agent state
  class AgentState(TypedDict):
    question: Annotated[str, 'The question to answer']
    attachments: Annotated[list[str], 'The attachments to the question']
    answer: Annotated[str, 'The answer to the question']

  # instantiate llm
  llm = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
  )

  # instantiate graph builder with state
  workflow = StateGraph(AgentState)

  # create a node to generate an answer from an LLM
  def answer(state: AgentState) -> AgentState:
    # use a Groq LLM to generate an answer
    # answer = llm.invoke(state['question'])
    answer = 'placeholder'
    return {
      'answer': answer.content
    }
  workflow.add_node('answer', answer)
  workflow.add_edge(START, 'answer')
  workflow.add_edge('answer', END)

  return workflow.compile()