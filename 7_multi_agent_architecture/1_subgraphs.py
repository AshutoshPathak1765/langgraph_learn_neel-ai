from typing import TypedDict, Annotated, List
from langgraph.graph import add_messages, StateGraph, END, START
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


class ChildState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


search_tool = TavilySearch(
    max_results=3,
    topic="general",
)

tools = [search_tool]

llm = init_chat_model("gpt-5.4-mini")

llm_with_tools = llm.bind_tools(tools=tools)


def agent(state: ChildState):
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
    }


tool_node = ToolNode(tools=tools)

child_graph = StateGraph(ChildState)

child_graph.add_node("agent", agent)
child_graph.add_node("tools", tool_node)

child_graph.add_conditional_edges("agent", tools_condition)
child_graph.add_edge("tools", "agent")

child_graph.set_entry_point("agent")

child_app = child_graph.compile()

# graph_png = child_app.get_graph(xray=True).draw_mermaid_png()

# with open("langgraph_flow.png", "wb") as f:
#     f.write(graph_png)

# response = child_app.invoke(
#     {"messages": [HumanMessage(content="How is the weather in Jamshedpur?")]}
# )
# print(response["messages"][-1].content)


# Define parent graph with the same schema
class ParentState(TypedDict):
    messages: Annotated[list, add_messages]


# Create parent graph
parent_graph = StateGraph(ParentState)

# Add the subgraph as a node
parent_graph.add_node("search_agent", child_app)
# Connect the flow
parent_graph.add_edge(START, "search_agent")
parent_graph.add_edge("search_agent", END)

# Compile parent graph
parent_app = parent_graph.compile()

# graph_png = parent_app.get_graph(xray=True).draw_mermaid_png()

# with open("langgraph_flow.png", "wb") as f:
#     f.write(graph_png)
# Run the parent graph
# response = parent_app.invoke(
#     {"messages": [HumanMessage(content="How is the weather in Jamshedpur?")]}
# )
# print(response["messages"][-1].content)


# Define parent graph with different schema
class QueryState(TypedDict):
    query: str
    response: str


# Function to invoke subgraph
def search_agent(state: QueryState):
    # Transform from parent schema to subgraph schema
    subgraph_input = {"messages": [HumanMessage(content=state["query"])]}

    # Invoke the subgraph
    subgraph_result = child_app.invoke(subgraph_input)

    # Transform response back to parent schema
    assistant_message = subgraph_result["messages"][-1]
    return {"response": assistant_message.content}


# Create parent graph
query_graph = StateGraph(QueryState)

query_graph.add_node("search_agent_subgraph", search_agent)

# Connect the flow
query_graph.add_edge(START, "search_agent_subgraph")
query_graph.add_edge("search_agent_subgraph", END)

# Compile parent graph
query_app = query_graph.compile()

graph_png = query_app.get_graph(xray=True).draw_mermaid_png()

with open("langgraph_flow.png", "wb") as f:
    f.write(graph_png)

# Run the query graph
response = query_app.invoke({"query": "How is the weather in Jamshedpur?", "response": ""})
print(response["response"])
