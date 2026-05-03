from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import config

load_dotenv()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = init_chat_model("gpt-5.4-mini")

GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"


def generate_post(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def get_review_decision(state: State):
    post_content = state["messages"][-1].content

    print("\n📢 Current LinkedIn Post:\n")
    print(post_content)
    print("\n")

    decision = input("Post to LinkedIn? (yes/no): ")

    if decision.lower() == "yes":
        return "post"
    else:
        return "feedback"


def post(state: State):
    final_post = state["messages"][-1].content
    print("\n📢 Final LinkedIn Post:\n")
    print(final_post)
    print("\n✅ Post has been approved and is now live on LinkedIn!")


def collect_feedback(state: State):
    feedback = input("How can I improve this post?")
    return {"messages": [HumanMessage(content=feedback)]}


graph = StateGraph(State)

graph.add_node(GENERATE_POST, generate_post)
graph.add_node(POST, post)
graph.add_node(COLLECT_FEEDBACK, collect_feedback)

graph.set_entry_point(GENERATE_POST)

graph.add_conditional_edges(
    GENERATE_POST, get_review_decision, {"post": POST, "feedback": COLLECT_FEEDBACK}
)
# graph.add_edge(POST, END)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

app = graph.compile()

graph_png = app.get_graph(xray=True).draw_mermaid_png()

with open("langgraph_flow.png", "wb") as f:
    f.write(graph_png)


response = app.invoke(
    {
        "messages": [
            HumanMessage(
                content="Write me a LinkedIn post on AI Agents taking over content creation"
            )
        ]
    }
)

print(response["messages"][-1].content)
