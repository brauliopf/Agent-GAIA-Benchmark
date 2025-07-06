from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .tools import wikipedia_search_tool, tavily_search_tool, audio_2_text, read_image
from .models import Plan, Act
import os

###########
#  Planner
###########
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
            If the task mentions an auxiliar file, let the agent know by setting true the flag 'has_file'. The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

planner_model = planner_prompt | ChatOpenAI( # The pipe operator chains the prompt to the next component (chat llm)
    model="gpt-4o", temperature=0.4
).with_structured_output(Plan)

###########
#  Executor
###########
llm = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model="deepseek-r1-distill-llama-70b",
    temperature=0.6,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=3,
  )

tools = [wikipedia_search_tool, tavily_search_tool, audio_2_text, read_image]
executor_prompt = "You are a helpful assistant."
executor_model = create_react_agent(llm, tools, prompt=executor_prompt)

###########
#  Replanner
###########
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

You must decide whether to return to the user or continue seeking the solution.

If you need more steps, then adjust the plan accordingly and respond with action Plan.
If no more steps are needed and you can return to the user, then create an answer and respond with action: Respond.

The answer must be crisp, clear and use proper grammar, with sentence case. If asked a number, return the number. If asked a translation, return the translation. If asked a list, return the list. Make no additional comments, greetings or explanations."""
)

replanner_model = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)