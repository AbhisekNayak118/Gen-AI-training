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

template2 = PromptTemplate(
    template='Summarize the following text in exactly 5 concise bullet points:\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

# --- USING LCEL FORMAT ---
# Chain 1 generates the detailed report and passes the string output
chain1 = template1 | model | parser

# Chain 2 takes the output of Chain 1 as "text", formats it into template2, runs the model, and parses the output
chain2 = {"text": chain1} | template2 | model | parser

# Invoke the final chain which automatically runs chain1 first
final_summary = chain2.invoke({'topic': 'black hole'})

print(final_summary)
