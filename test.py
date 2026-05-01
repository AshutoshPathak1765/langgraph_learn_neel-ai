from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilySearch(max_results=5, topic="general")
response=search_tool.invoke("Latest AI news")
print(response["results"][0]["content"])