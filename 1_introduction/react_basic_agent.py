from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
import datetime

load_dotenv()

search_tool = TavilySearch(
    max_results=5,
    topic="general",
)


def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


model = init_chat_model("gpt-5-mini")

system_prompt = """
You are a smart and concise assistant.

Use available tools whenever they help improve accuracy or provide real-time information.

Rules:
- Give direct answers first.
- Keep responses short and clear unless more detail is requested.
- Summarize tool results cleanly instead of repeating raw outputs.
- For weather, time, or live info, respond practically and briefly.
- For casual messages, reply naturally.
- If unsure, say so honestly.
- Avoid unnecessary follow-up questions.
"""
agent = create_agent(
    model=model,
    tools=[search_tool, get_system_time],
    system_prompt=system_prompt,
    debug=True,
)

while True:
    query = input("👨🏻:")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Exiting the chat. Goodbye!")
        break

    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print(f"🤖:{response['messages'][-1].content}")