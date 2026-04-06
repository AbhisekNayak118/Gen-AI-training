import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:groq",
    messages=[
        {
            "role": "user",
            "content": "Write a poem on Lucifer, the fallen angel."
        }
    ],
)

print(completion.choices[0].message.content)