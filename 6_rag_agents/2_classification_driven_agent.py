from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from typing import Annotated, Literal, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import os
from langgraph.graph import StateGraph, END, add_messages
from pydantic import BaseModel, Field
import psycopg
from urllib.parse import quote_plus
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

pg_password = quote_plus(os.getenv("POSTGRES_PASSWORD"))

conn = psycopg.connect(
    f"postgresql://postgres:{pg_password}@db.zuydbwhirdgeqoteqbiy.supabase.co:5432/postgres",
    sslmode="require",
)

conn.autocommit = True

memory = PostgresSaver(conn=conn)

# memory.setup()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "chroma_langchain_db")
print(f"Persistent directory path: {persistent_directory}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """ 
Answer the question based only on the following context: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = init_chat_model("gpt-5.4-mini")

rag_chain = prompt | llm


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    on_topic: str


class GradeQuestion(BaseModel):
    """Boolean value to check whether a question is related to the Peak Performance Gym"""

    score: Annotated[
        Literal["Yes", "No", None],
        Field(description="Question is about gym? If yes -> 'Yes' if not -> 'No' "),
    ]


def question_classifier(state: AgentState):
    question = state["messages"][-1].content
    system = """ You are a classifier that determines whether a user's question is about one of the following topics 
    
    1. Gym History & Founder
    2. Operating Hours
    3. Membership Plans 
    4. Fitness Classes
    5. Personal Trainers
    6. Facilities & Equipment
    
    If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'.

    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system),
            HumanMessage(content=f"User question:{question}"),
        ]
    )
    llm_with_structured_output = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | llm_with_structured_output
    response = grader_llm.invoke({"question": question})
    state["on_topic"] = response.score
    return state


def on_topic_router(state: AgentState):
    on_topic = state["on_topic"]
    if on_topic.lower() == "yes":
        return "on_topic"
    else:
        return "off_topic"


def retrieve(state: AgentState):
    question = state["messages"][-1].content
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state


def generate_answer(state: AgentState):
    question = state["messages"][-1].content
    documents = state["documents"]
    docs = format_docs(docs=documents)
    generation = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [generation]}


def off_topic_response(state: AgentState):
    return {
        "messages": [SystemMessage(content="I'm sorry! I cannot answer this question!")]
    }


workflow = StateGraph(AgentState)
workflow.add_node("topic_decision", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_answer", generate_answer)

workflow.add_conditional_edges(
    "topic_decision",
    on_topic_router,
    {"on_topic": "retrieve", "off_topic": "off_topic_response"},
)

workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("off_topic_response", END)

workflow.set_entry_point("topic_decision")

app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": 1}}

graph_png = app.get_graph(xray=True).draw_mermaid_png()

with open("langgraph_flow.png", "wb") as f:
    f.write(graph_png)

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
