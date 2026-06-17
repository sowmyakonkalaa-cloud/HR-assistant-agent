import os
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# Page config
st.set_page_config(page_title="HR Assistant Agent", page_icon="🤖")
st.title("🤖 HR Assistant Agent")
st.caption("Ask me anything about HR policies, employee data, or general questions!")

# Load API key from Streamlit secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Initialize everything once using cache
@st.cache_resource
def load_agent():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Load PDF
    loader = PyPDFLoader("07_pay_benefits_and_leave_policy.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Tool 1 - PDF
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    def search_hr_policy(query):
        docs = retriever.get_relevant_documents(query)
        return "\n".join([doc.page_content for doc in docs])
    pdf_tool = Tool(
        name="HR Policy Search",
        func=search_hr_policy,
        description="Use for questions about HR policies, leave, salary, WFH rules.")

    # Tool 2 - CSV
    df = pd.read_csv("employee_data.csv")
    def search_employee_data(query):
        return f"Employee Data:\n{df.to_string()}\n\nQuestion: {query}"
    csv_tool = Tool(
        name="Employee Data Search",
        func=search_employee_data,
        description="Use for questions about employee leave balance, department, leaves taken.")

    # Tool 3 - Web
    search = DuckDuckGoSearchRun()
    web_tool = Tool(
        name="Web Search",
        func=search.run,
        description="Use for current news or general questions not related to HR or employee data.")

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create agent
    agent = initialize_agent(
        tools=[pdf_tool, csv_tool, web_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )
    return agent

# Load agent
agent = load_agent()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.run(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
