from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv

load_dotenv()

memory = MemorySaver()

search_tool = TavilySearch(max_results=3, topic="general")

tools = [search_tool]

tool_node = ToolNode(tools=tools)

llm = init_chat_model("gpt-5.4-mini")
llm_with_tools = llm.bind_tools(tools)

ASSISTANT = "assistant"
TOOLS = "tools"


class BasicState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def assistant(state: BasicState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(BasicState)
graph.add_node(ASSISTANT, assistant)
graph.add_node(TOOLS, tool_node)
graph.set_entry_point(ASSISTANT)

graph.add_conditional_edges(ASSISTANT, tools_condition)

graph.add_edge(TOOLS, ASSISTANT)

app = graph.compile(checkpointer=memory, interrupt_before=[TOOLS])

graph_png = app.get_graph(xray=True).draw_mermaid_png()

with open("langgraph_flow.png", "wb") as f:
    f.write(graph_png)

config = {"configurable": {"thread_id": 1}}

events = app.stream(
    {"messages": [HumanMessage(content="What is the current weather in Jamshedpur?")]},
    config=config,
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()

snapshot = app.get_state(config=config)
print(snapshot.next)

events = app.stream(None, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()
