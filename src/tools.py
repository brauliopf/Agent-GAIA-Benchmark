from typing import Annotated
from langchain_community.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

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