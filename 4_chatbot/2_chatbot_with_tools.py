from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

search_tool = TavilySearch(max_results=3, topic="general")


def stock_price_tool(query: str) -> str:
    """Fetches the latest stock price for a given query."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={query}&interval=5min&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"

    r = requests.get(url)
    data = r.json()
    return data


llm = init_chat_model("gpt-5.4")

tools = [search_tool, stock_price_tool]

llm_with_tools = llm.bind_tools(tools=tools)


class BasicChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state: BasicChatState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph = StateGraph(BasicChatState)

tool_node = ToolNode([search_tool, stock_price_tool])

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")
app = graph.compile()

graph_png = app.get_graph(xray=True).draw_mermaid_png()

with open("langgraph_flow.png", "wb") as f:
    f.write(graph_png)


while True:
    query = input("👨🏻:")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Exiting the chat. Goodbye!")
        break

    response = app.invoke({"messages": [HumanMessage(content=query)]})
    print(f"🤖:{response['messages'][-1].content}")
