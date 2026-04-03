from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm_openai = OpenAI(model = "gpt-3.5-turbo")

result = llm_openai("What is the best indian food ? ")
print(result )

