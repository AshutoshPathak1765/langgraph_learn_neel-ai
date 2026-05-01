from react_state import ChatState
from react_tools import *
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

tools = [search_tool, calculator]

llm = init_chat_model("gpt-5.4")
llm_with_tools = llm.bind_tools(tools)


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)
