from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gpt-5.4")


class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: BasicChatState):
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

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
