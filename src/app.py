import os
import pandas as pd
import requests
import gradio as gr
from agent import AgentState, build_graph

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Agent Definition ---
class SmartyAgent:
    def __init__(self):
        print("Agent initialized.")
        self.agent = build_graph()
    # __call__ turns an instance of SmartyAgent into a callable object
    def __call__(self, question: str, task_id: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        response = self.agent.invoke(AgentState({'question':question, 'task_id': task_id}))
        return response['answer']

def fetch_questions_for_selection():
    """
    Fetches all questions and returns them formatted for selection interface.
    Returns updated choices for CheckboxGroup and status message.
    """
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        
        if not questions_data:
            return gr.CheckboxGroup(choices=[], value=[]), "No questions available."
        
        # Format questions for checkbox display
        question_choices = []
        for item in questions_data:
            task_id = item.get("task_id")
            question_text = item.get("question", "")
            if task_id and question_text:
                # Create display label with task_id and first 50 chars of question
                display_label = f"{task_id}: {question_text}"
                question_choices.append((display_label, task_id))
        
        return gr.CheckboxGroup(choices=question_choices, value=[]), f"Loaded {len(question_choices)} questions."
    
    except Exception as e:
        return gr.CheckboxGroup(choices=[], value=[]), f"Error fetching questions: {e}"

def run_and_submit_test(selected_questions):
    """
    Fetches questions, runs the BasicAgent on selected questions only, and displays the result.
    """
    if not selected_questions:
        return "No questions selected. Please select at least one question to run the test.", pd.DataFrame()

    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = SmartyAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", pd.DataFrame()
    
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", pd.DataFrame()
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", pd.DataFrame()
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", pd.DataFrame()

    # 3. Run your Agent on selected questions only
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(selected_questions)} selected questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        # Only process if this question was selected
        if task_id not in selected_questions:
            continue
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            # >>>>>>>> HERE <<<<<<<<
            # __call__ formats the input parameters as the Agent State
            submitted_answer = agent(question_text, task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    return f"Agent finished running on {len(results_log)} selected questions.", pd.DataFrame(results_log)

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = SmartyAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text, task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

def submit_question(question: str):
    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = SmartyAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", pd.DataFrame()
    
    answer = agent(question, "000")
    return answer

# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    
    gr.Markdown("# Basic Agent Evaluation Runner")

    # HF Integration - Display only in development mode
    if os.getenv("ENV_MODE") == "DEV":
        gr.Markdown("## Automatic Evaluation")
        
        gr.Markdown(
            """
            **Instructions:**

            1.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
            2.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

            ---
            **Disclaimers:**
            Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go thro ugh all the questions).
            This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
            """
        )

        gr.LoginButton()
    
    gr.Markdown("## Run Evaluation & Submit All Answers")
    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

    gr.Markdown("## Make your own question")
    # add a textbox for the user to input a question
    question_input = gr.Textbox(label="Question", lines=1, interactive=True)
    # add a button to submit the question
    submit_question_button = gr.Button("Submit Question", variant="primary")
    # add a textbox to display the question
    question_output = gr.Textbox(label="Question", lines=1, interactive=False)
    

    # wire up the interactions
    submit_question_button.click(fn=submit_question, inputs=[question_input], outputs=[question_output])

    gr.Markdown("---")

    gr.Markdown("## Test Evaluation")
    gr.Markdown("Select specific questions to test your agent on:")

    # Question selection interface
    load_questions_button = gr.Button("Load Questions", variant="secondary")
    question_status = gr.Textbox(label="Question Load Status", lines=1, interactive=False)
    question_checkboxes = gr.CheckboxGroup(
        label="Select Questions to Test",
        choices=[],
        value=[],
        interactive=True
    )
    
    run_test_button = gr.Button("Run Test Evaluation on Selected Questions", variant="primary")

    status_output_test = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table_test = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    # Wire up the interactions
    load_questions_button.click(
        fn=fetch_questions_for_selection,
        outputs=[question_checkboxes, question_status]
    )
    
    run_test_button.click(
        fn=run_and_submit_test,
        inputs=[question_checkboxes],
        outputs=[status_output_test, results_table_test]
    )


if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    
    # Get port from environment variable (Render sets this automatically)
    port = int(os.getenv("PORT", 7860))  # Default to 7860 if PORT not set
    
    demo.launch(
        debug=True, 
        share=False,
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=port        # Use the port from environment or default
    )