import os
import requests
import urllib.request
import tempfile
from typing import Annotated
from langchain_community.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch
import json
from groq import Groq
import base64
from openai import OpenAI
import subprocess
import sys
import pandas as pd
import openpyxl
import google.generativeai as genai
from dotenv import load_dotenv
import operator

load_dotenv()

# Add tool to download a file from a url and save it to a local path in a temporary file
@tool
def download_file_tool(task_id: str) -> str:
    """
    Downloads a file from the web and stores it in a temporary file. Returns the absolute path for the file
    Args:
      task_id (str): the task_id of the file to download.
    Returns:
      file_path (str): the absolute path to the file.
    """

    # Build the URL and make a GET request to determine content-type
    url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"
    
    # Map common content types to extensions
    content_type_map = {
        'application/pdf': '.pdf',
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg', 
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'audio/mpeg': '.mp3',
        'audio/mp3': '.mp3',
        'audio/wav': '.wav',
        'audio/ogg': '.ogg',
        'video/mp4': '.mp4',
        'video/webm': '.webm',
        'video/avi': '.avi',
        'text/plain': '.txt',
        'text/csv': '.csv',
        'application/json': '.json',
        'application/xml': '.xml',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.ms-excel': '.xls',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/zip': '.zip',
        'application/x-zip-compressed': '.zip',
    }
    
    try:
        # Make GET request to get content-type and download the file
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        # Try to get extension from content-type
        extension = ''
        for ct, ext in content_type_map.items():
            if ct in content_type:
                extension = ext
                break
        
        # Create base filename from task_id
        base_name = task_id
            
    except:
        # Fallback if request fails
        extension = ''
        base_name = 'download'
        response = None

    # Create temporary file with detected extension
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        prefix=f'tmp_{base_name}_',
        suffix=extension
    )
    temp_file_path = temp_file.name
    temp_file.close()

    # Save the file content
    if response is not None:
        # We already have the content from the GET request above
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)
    else:
        # Fallback: use urllib if the requests approach failed
        urllib.request.urlretrieve(url, temp_file_path)

    print(f"Downloaded file to: {temp_file_path}")

    return temp_file_path

@tool
def wikipedia_search_tool(
  query: Annotated[str, 'The query to search Wikipedia for']
) -> str:
  "Perform a search on Wikipedia"
  print(f">>>>> Searching Wikipedia for: {query}")
  return WikipediaAPIWrapper().run(query)

@tool
def tavily_search_tool(
  query: Annotated[str, 'The query to search Tavily for']
) -> str:
  "Perform a search on Tavily"
  print(f">>>>> Searching Tavily for: {query}")
  return TavilySearch(max_results=3).run(query)

@tool
def audio_2_text(file_path: str) -> str:
    """
    transcribe an audio file to text
    Args:
      file_path (str): the absolute file_path of the targeted audio.
    """

    # Initialize the Groq client
    client = Groq()

    # Open the audio file
    with open(file_path, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
          file=file, # Required audio file
          model="whisper-large-v3-turbo", # Required model to use for transcription
          prompt="Specify context or spelling",  # Optional
          response_format="verbose_json",  # Optional
          timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
          # language="en",  # Optional
          temperature=0.0  # Optional
        )
        # To print only the transcription text, you'd use print(transcription.text
        # Print the entire transcription object to access timestamps
    
    return json.dumps(transcription.text, indent=2, default=str)


def encode_image(image_path: str) -> str:
    """
    Encode an image to base64
    Args:
      image_path (str): the absolute file_path of the targeted image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@tool
def read_image(image_path: str) -> str:
    """
    Read an image and return the text description.
    Args:
      image_path (str): the state variable 'attachment' has the absolute file_path of the targeted image.
    """
    base64_image = encode_image(image_path)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image? Describe the image in detail. If this is a board game, read the current status of the board, but do not make an analysis at all"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="gpt-4.1-2025-04-14"
    )

    return chat_completion.choices[0].message.content


def code_executor(code: str, timeout: int = 100) -> dict:
    """
    Executes Python code in a subprocess and returns the result.
    """
    result = {"stdout": "", "stderr": "", "exit_code": 0}
    try:
        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        result.update({
            "stdout": process.stdout,
            "stderr": process.stderr,
            "exit_code": process.returncode,
        })
    except subprocess.TimeoutExpired:
        result["stderr"] = "Execution timed out."
    except Exception as e:
        result["stderr"] = str(e)
    return result

@tool
def execute_code_from_file(file_path: str, timeout: int = 10) -> dict:
    """
    Reads Python code from a file and executes it in a subprocess.
    
    Args:
        file_path (str): Path to the Python file to execute
        timeout (int): Maximum execution time in seconds
        
    Returns:
        dict: Execution results including stdout, stderr, and exit code
    """
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        result = code_executor(code)
        return result["stdout"] if result["stdout"] else 0
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"File not found: {file_path}", "exit_code": 1}
    except Exception as e:
        return {"stdout": "", "stderr": f"Error reading file: {str(e)}", "exit_code": 1}

@tool
def read_excel_file(file_path: str) -> str:
    """
    Read an Excel file and return a CSV string.
    Args:
      file_path (str): the absolute file_path of the targeted Excel file.
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Return the text description
    return df.to_csv(index=False)

@tool
def calculator(term1: str, term2: str, operation: str) -> str:
    """
    Calculate the result of a mathematical operation between two terms.
    
    Args:
      term1 (str): the first term (must be a valid number).
      term2 (str): the second term (must be a valid number).
      operation (str): the operation to perform. Supported operations:
        - '+' : addition
        - '-' : subtraction
        - '*' : multiplication
        - '/' : division
        - '//' : floor division
        - '%' : modulo
        - '**' : exponentiation
        - '==' : equality comparison
        - '!=' : inequality comparison
        - '<' : less than
        - '<=' : less than or equal
        - '>' : greater than
        - '>=' : greater than or equal
    
    Returns:
      str: the result of the operation as a string.
    
    Raises:
      ValueError: if the operation is not supported or if terms are not valid numbers.
    """
    # Safe operators mapping
    safe_operators = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '//': operator.floordiv,
        '%': operator.mod,
        '**': operator.pow,
        '==': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge,
    }
    
    if operation not in safe_operators:
        raise ValueError(f"Unsupported operation: {operation}. Supported operations: {list(safe_operators.keys())}")
    
    try:
        # Convert terms to float for arithmetic operations
        num1 = float(term1)
        num2 = float(term2)
        
        # Perform the operation
        result = safe_operators[operation](num1, num2)
        
        # Return as string, preserving integer format for whole numbers
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        else:
            return str(result)
            
    except ValueError as e:
        raise ValueError(f"Invalid numeric terms: term1='{term1}', term2='{term2}'. Error: {e}")
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed")
    except Exception as e:
        raise ValueError(f"Calculation error: {e}")

@tool
def query_video(video_url: str, query: str) -> str:
    """
    Query a video and return the answer to a question.
    Args:
      video_url (str): the YouTube URL of the video to query.
      query (str): the question to ask about the video.
    """

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Create a model instance
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Generate content
    response = model.generate_content([
        {
            "parts": [
                {
                    "file_data": {
                        "file_uri": video_url
                    }
                },
                {
                    "text": f"""Think step-by-step and respond to the following question about the video content. Provide a structured output with the following fields: 'answer', 'reasoning'. For example: {{'answer': 'The answer to the question', 'reasoning': 'The reasoning for the answer'}}.
                    This is the question: {query}"""
                }
            ]
        }
    ])

    return response.text