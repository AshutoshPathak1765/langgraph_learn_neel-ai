from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from langchain_core.tools.retriever import create_retriever_tool
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
import psycopg
from urllib.parse import quote_plus
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_tavily import TavilySearch

load_dotenv()

pg_password = quote_plus(os.getenv("POSTGRES_PASSWORD"))

conn = psycopg.connect(
    f"postgresql://postgres:{pg_password}@db.zuydbwhirdgeqoteqbiy.supabase.co:5432/postgres",
    sslmode="require",
)

conn.autocommit = True

memory = PostgresSaver(conn=conn)

# memory.setup()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "chroma_langchain_db")

docs = [
    Document(
        page_content="Peak Performance Gym was founded in 2015 by former Olympic athlete Marcus Chen. With over 15 years of experience in professional athletics, Marcus established the gym to provide personalized fitness solutions for people of all levels. The gym spans 10,000 square feet and features state-of-the-art equipment.",
        metadata={"source": "about.txt"},
    ),
    Document(
        page_content="Peak Performance Gym is open Monday through Friday from 5:00 AM to 11:00 PM. On weekends, our hours are 7:00 AM to 9:00 PM. We remain closed on major national holidays. Members with Premium access can enter using their key cards 24/7, including holidays.",
        metadata={"source": "hours.txt"},
    ),
    Document(
        page_content="Our membership plans include: Basic (₹1,500/month) with access to gym floor and basic equipment; Standard (₹2,500/month) adds group classes and locker facilities; Premium (₹4,000/month) includes 24/7 access, personal training sessions, and spa facilities. We offer student and senior citizen discounts of 15% on all plans. Corporate partnerships are available for companies with 10+ employees joining.",
        metadata={"source": "membership.txt"},
    ),
    Document(
        page_content="Group fitness classes at Peak Performance Gym include Yoga (beginner, intermediate, advanced), HIIT, Zumba, Spin Cycling, CrossFit, and Pilates. Beginner classes are held every Monday and Wednesday at 6:00 PM. Intermediate and advanced classes are scheduled throughout the week. The full schedule is available on our mobile app or at the reception desk.",
        metadata={"source": "classes.txt"},
    ),
    Document(
        page_content="Personal trainers at Peak Performance Gym are all certified professionals with minimum 5 years of experience. Each new member receives a complimentary fitness assessment and one free session with a trainer. Our head trainer, Neha Kapoor, specializes in rehabilitation fitness and sports-specific training. Personal training sessions can be booked individually (₹800/session) or in packages of 10 (₹7,000) or 20 (₹13,000).",
        metadata={"source": "trainers.txt"},
    ),
    Document(
        page_content="Peak Performance Gym's facilities include a cardio zone with 30+ machines, strength training area, functional fitness space, dedicated yoga studio, spin class room, swimming pool (25m), sauna and steam rooms, juice bar, and locker rooms with shower facilities. Our equipment is replaced or upgraded every 3 years to ensure members have access to the latest fitness technology.",
        metadata={"source": "facilities.txt"},
    ),
]

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persistent_directory,
    collection_name="gym_collection",
)

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="rag_retreiver_tool",
    description="Information related to Gym History & Founder, Operating Hours, Membership Plans, Fitness Classes, Personal Trainers, and Facilities & Equipment of Peak Performance Gym",
)

search_tool = TavilySearch(max_results=3, topic="general")

tools = [retriever_tool, search_tool]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def agent(state: AgentState):
    messages = state["messages"]
    model = init_chat_model("gpt-5.4-mini")
    model_with_tools = model.bind_tools(tools=tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(AgentState)
tool_node = ToolNode(tools=tools)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=memory)

graph_png = app.get_graph(xray=True).draw_mermaid_png()

with open("langgraph_flow.png", "wb") as f:
    f.write(graph_png)

    config = {"configurable": {"thread_id": 1}}

while True:
    query = input("👨🏻:")

    if query.lower() in ["exit", "quit", "bye"]:
        print("Exiting the chat. Goodbye!")
        break
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "documents": [],
        "on_topic": None,
    }
    response = app.invoke(initial_state, config=config)
    print(f"🤖:{response["messages"][-1].content}")
