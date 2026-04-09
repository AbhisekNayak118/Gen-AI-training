import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import time

# Force API key (from notebook)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBjAL10Otx8iSpPGJ5V5XOzIxjmXjsIxrk"

st.set_page_config(page_title="Financial Advisor", page_icon="📈", layout="centered")

st.title("📈 AI Financial Advisor")
st.write("Upload your bank statement and ask questions about your finances.")

tmp_file_path = None

uploaded_file = st.file_uploader("Choose a bank statement PDF", type="pdf")
if uploaded_file is not None:
    # Save uploaded file temporarily to use PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    st.success("PDF Uploaded Successfully!")

if tmp_file_path:
    # Use session state to cache the components so we don't recreate them
    if "rag_setup_done" not in st.session_state or st.session_state.get("pdf_path") != tmp_file_path:
        with st.spinner("Processing document..."):
            try:
                # 2. Load the Bank Statement PDF
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()

                # 3. Chunk the PDF
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700, 
                    chunk_overlap=150
                )
                chunks = text_splitter.split_documents(docs)

                # 4. Create the Vector Database
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

                # Set up the retriever
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                # 5. Set up Gemini and the Financial Advisor Persona
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2) 

                template = """You are a professional, highly analytical, and helpful Financial Advisor.
                            Your client has provided you with their bank statement.
                            Use ONLY the provided context (which contains extracts from their statement) to answer their question.
                            Do not make up any transactions, balances, or financial advice that isn't directly supported by the context.
                            If the context doesn't contain the answer, politely tell the client that the information is missing from the provided statement.

                            Bank Statement Context: 
                            {context}

                            Client Question: {question}

                            Advisor Response:"""
                prompt = PromptTemplate.from_template(template)

                def format_docs(retrieved_docs):
                    return "\n\n".join(doc.page_content for doc in retrieved_docs)

                st.session_state.retriever = retriever
                st.session_state.format_docs = format_docs
                st.session_state.prompt = prompt
                st.session_state.llm = llm
                
                st.session_state.rag_setup_done = True
                st.session_state.pdf_path = tmp_file_path
                st.session_state.messages = [] # Reset messages when new PDF is loaded
                
            except Exception as e:
                st.error(f"Error processing the PDF: {e}")
                st.stop()
    
    # 6. Chat Interface
    st.subheader("Ask your Financial Advisor")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if question := st.chat_input("e.g. What was my largest debit transaction this month?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.status("Processing your query...", expanded=True) as status:
                try:
                    st.write("🔍 Retrieving relevant context from bank statement...")
                    docs = st.session_state.retriever.invoke(question)
                    time.sleep(2)
                    
                    st.write("⚙️ Formatting context...")
                    context = st.session_state.format_docs(docs)
                    time.sleep(2)
                    
                    st.write("🧠 Generating AI response...")
                    prompt_value = st.session_state.prompt.invoke({"context": context, "question": question})
                    response_msg = st.session_state.llm.invoke(prompt_value)
                    
                    # LLM invoke returns an AIMessage, we extract its content
                    response = response_msg.content
                    
                    status.update(label="Response generated!", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="Error occurred", state="error", expanded=True)
                    st.error(f"An error occurred: {e}")
                    response = None

            if response:
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload a bank statement PDF to proceed.")
