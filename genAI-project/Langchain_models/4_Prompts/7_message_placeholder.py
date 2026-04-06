from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
# from langchain_core.messages import HumanMessage
import os

# chat template
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', "{query}")
])

# load chat history
chat_history=[]
if os.path.exists("chat_history.txt"):
    with open("chat_history.txt", "r") as f:
        chat_history.extend(f.readlines())
else:
    print("No chat history found. Starting fresh.")
print(chat_history)

# Create prompt
prompt = chat_template.invoke(
    {
        'chat_history': chat_history, 
        'query': "What is your return policy?"
    })

print(prompt)