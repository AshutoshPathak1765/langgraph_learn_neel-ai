from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, Annotated
from chains import generation_chain, reflection_chain
import operator

load_dotenv()


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def generate_node(state: MessagesState):
    response = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}


def reflect_node(state: MessagesState):
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}


REFLECT = "reflect"
GENERATE = "generate"

graph = StateGraph(MessagesState)
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)


def should_continue(state: MessagesState):
    if len(state["messages"]) > 6:
        return "stop"
    return "continue"


graph.add_conditional_edges(
    GENERATE, should_continue, {"continue": REFLECT, "stop": END}
)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

graph_png = app.get_graph(xray=True).draw_mermaid_png()

with open("langgraph_flow.png", "wb") as f:
    f.write(graph_png)

initial_state = {
    "messages": [HumanMessage(content="AI Agents taking over content creation")]
}

response = app.invoke(initial_state)

print(response["messages"][-1].content)
