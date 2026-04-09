from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import os
load_dotenv()

llm = HuggingFaceEndpoint(
    model="MiniMaxAI/MiniMax-M2.5",
    task = "text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm = llm)
# model = llm

parser = JsonOutputParser()


# Template -> Initial prompt template tp generate a detailed report on a topic
t1 = PromptTemplate(
    template='Write a detailed report on {topic}. \n {format_instruction}',
    input_variables=['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

chain = t1 | model | parser
# print(parser.get_format_instructions())
# print("----------------------------------------------------------------", "\n",  chain, " \n", "----------------------------------------------------------------")
res = chain.invoke({'topic': "Ultron from Marvel Cinematic Universe"})

print(res)