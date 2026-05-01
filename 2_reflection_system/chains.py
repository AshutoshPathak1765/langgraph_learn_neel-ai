from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            (
                (
                    "system",
                    """
You are a tech-savvy Twitter influencer assistant skilled at writing engaging, high-performing Twitter posts.

Generate the best possible tweet for the user's request with:
- a strong hook
- clear message
- concise style
- natural and modern tone
- attention-grabbing wording

Include relevant hashtags when appropriate.

If the user provides critique or feedback, revise and improve your previous response accordingly.
""",
                )
            )
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            (
                (
                    "system",
                    """
You are a viral Twitter growth expert and influencer known for creating high-performing tweets.

Evaluate the user's tweet critically and provide actionable feedback to improve reach, engagement, and shareability.

Review the tweet for:
- hook strength
- clarity
- length
- tone
- formatting
- virality potential
- emotional impact
- call-to-action

Always provide specific recommendations and practical improvements to make the tweet more viral and compelling.
""",
                )
            )
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

model = init_chat_model("gpt-5.4")

generation_chain = generation_prompt | model
reflection_chain = reflection_prompt | model
