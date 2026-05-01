from langchain_core.messages import HumanMessage

from react_state import ChatState
from langgraph.graph import StateGraph
from react_nodes import chat_node, tool_node
from langgraph.prebuilt import tools_condition

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

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
