from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful {domain} expert"),
    ('human', "Explain in simple words, what is {input}?"),
])  

prompt = chat_template.invoke({"domain": 'Movies', "input" : 'Iron Man' })

print(prompt)