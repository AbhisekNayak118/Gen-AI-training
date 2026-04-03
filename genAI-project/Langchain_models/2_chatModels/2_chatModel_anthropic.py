from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
load_dotenv()

chatModel2 = ChatAnthropic(model_name = 'Claude Sonnet 4.6', timeout=10, stop=None, temperature=1.5)
result = chatModel2.invoke('Write a poem on Lucifer, the fallen angel')


