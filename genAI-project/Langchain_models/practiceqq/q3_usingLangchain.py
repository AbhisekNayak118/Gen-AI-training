from langchain_openai import OpenAiEmbeddings ,ChatOpenAI
from langchain_community.document_loader import TextLoader
from langchain_community.vector_store import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parser StrOutputParser
 
loader = TextLoader("data.text")
document = loader.load()
 
embedding = OpenAiEmbeddings(
    model="text-embedding-3-small",
    api_key ="api_key"
    )
vectorstore =FAISS.from_document(docuent,embeddings)
search = vectorstore.as_retriever(search_kwargs={"k":2})
 
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key ="api_key"
    )
 
prompt =ChatPromptTemplate(
    "answer \n\n context:{context} \n\n questions:{questions}")
query = input("what is ai")
docs = search.invoke(query)
 
context = "\n.join([doc.page-content for doc in docs])"
 
chain = prompt |llm | strOutputParser
 
response = chain.invoke({
    "context":context,
    "query":query
})
 
print("answer",response)