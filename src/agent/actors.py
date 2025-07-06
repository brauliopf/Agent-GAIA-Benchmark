from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .tools import wikipedia_search_tool, tavily_search_tool, audio_2_text, read_image
from .models import Plan, Act, FinalAnswer
import os

#################################
#  Planner
#################################
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

#################################
#  Executor
#################################
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

task_prompt_template = PromptTemplate(
    input_variables=["plan_str", "task", "task_id"],
    template="""For the following plan:
{plan_str}

You are tasked with executing step 1, {task}. (task_id: {task_id})"""
)

#################################
#  Replanner
#################################
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
If no more steps are needed and you can return to the user, then create an answer and respond with action: Respond."""
)

replanner_model = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)

#################################
#  Final Answer
#################################
final_answer_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. You are given a question and an answer. You must return the final answer to the question.
    The final answer must be crisp, clear and use proper grammar, with sentence case. If asked a value, return the value. If asked a list, return the list. Make no additional comments, greetings or explanations.
    For example:

    Question: what is the opposite of up?
    Answer: The opposite of up is down.
    Final Answer: Down

    Question: Hi friend, how are you?
    Answer: I am fine. Thank you for asking.
    Final Answer: Fine

    Question: Hey there, your turn. What's the best move?
    Answer: The optimal move in this position it Re6.
    Final Answer: Re6

    Your turn:
    Question: {question}
    Answer: {answer}
    Final Answer: """
)

final_answer_model = final_answer_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(FinalAnswer)