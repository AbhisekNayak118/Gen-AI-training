from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from openai import chat
load_dotenv()

# Temparature is a parameter that controls the randomness of the model's output. A higher temperature will result in more random and creative responses, while a lower temperature will produce more focused and deterministic responses. Temperature value range for gemini = 0.0 to 1.0
# Max tokens is a parameter that limits the maximum number of tokens (words or subwords) in the generated response. Setting a lower max tokens value can help to keep the response concise and prevent it from being too long.

chatModel1 = ChatGoogleGenerativeAI(
    model = 'gemini-1.5-flash', 
    temperature=0.9, 
    max_tokens=10
    )
try:
    res = chatModel1.invoke("Write a short poem on Lucifer, the fallen angel.")
    print("\n--- Model Output ---\n")
    print(res.content)
except Exception as e:
    print(f"An error occurred: {e}")

