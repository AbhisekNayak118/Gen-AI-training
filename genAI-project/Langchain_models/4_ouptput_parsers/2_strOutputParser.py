from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

llm = HuggingFaceEndpoint(
    model="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),

    # max_new_tokens=512,
    temperature=0.7,
    # do_sample=True,
    # top_p=0.95,
    repetition_penalty=1.1,
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed technical report on {topic}. Include explanation, examples and applications.',
    input_variables=['topic']
)
# # 2nd prompt -> summerize the report in 5 bullets points
template2 = PromptTemplate(
    template='Summarize the following text in exactly 5 concise bullet points:\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

# --- USING LCEL FORMAT ---
chain = template1 | model | parser | template2 | model  | parser 

final_summary = chain.invoke({'topic': 'Lucifer Morningstar'})

print(final_summary)
