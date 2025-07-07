from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .tools import wikipedia_search_tool, tavily_search_tool, audio_2_text, read_image, execute_code_from_file, read_excel_file, calculator, query_video
from .models import Plan, Act, FinalAnswer
import os

#################################
#  Planner
#################################
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a skilled business analyst. For the given objective, come up with a sequence of milestones that will lead to the final answer. Each milestone can be achieved by executing a task (e.g.analyze an image, execute a code script, watch a video, download content from a URL, read a paper). The milestones should be in the correct order to lead to the final answer. Do not add any superfluous steps.
            If the task mentions an auxiliar file, let the agent know by setting true the flag 'has_file'. The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)

# The pipe operator chains the prompt to the next component (chat llm)
# langchain processes the object passed to planner_model and passes planner_promot just what it needs
planner_model = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0.4
).with_structured_output(Plan)

#################################
#  Executor
#################################
task_prompt_template = PromptTemplate(
    input_variables=["plan_str", "task", "task_id"],
    template="""
    User request: {objective}.
    
    To respond to this request, the team created the following plan:
    {plan_str}

    Follow this plan to respond to the user's request. Call tools as needed. Return when you have a final answer or you have completed the plan. (task_id: {task_id})

    The output from executed steps is in the message history.
    """
)

llm = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model="deepseek-r1-distill-llama-70b",
    temperature=0.3,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=3,
  )

tools = [wikipedia_search_tool, tavily_search_tool, audio_2_text, read_image, execute_code_from_file, read_excel_file, calculator, query_video]
executor_prompt = "You are a helpful assistant."
executor_model = create_react_agent(llm, tools, prompt=executor_prompt)

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
If you've reached a final answer, then create an answer and respond with action: Respond."""
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

    Question: Hi friend, where are you traveling to tomorrow?
    Answer: I am traveling to St. Petersburg.
    Final Answer: Saint Petersburg

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