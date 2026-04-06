from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

import os

load_dotenv()
model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_API_KEY")
)

messages = [
    SystemMessage(content="You are a helpful assistant that summarizes research papers."),
    HumanMessage(content="Please tell me about langChain.")
]


result = model.invoke(messages)
messages. append(AIMessage(content = result.content))

print(messages)