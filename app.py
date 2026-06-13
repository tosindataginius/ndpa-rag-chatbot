import os
import uuid
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')

load_dotenv()

st.set_page_config(
    page_title="NDPA Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stChatMessage { border-radius: 12px; }
    .block-container { padding-top: 1rem; max-width: 1200px; }
    .hero-card { padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #0f172a, #1e293b); color: white; margin-bottom: 20px; }
    .source-box { padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown("""
    <div class="hero-card">
        <h1>⚖️ Nigeria Data Protection Act AI Agent 🤖</h1> 
        <p>Ask me questions about NDPA and How to apply to your business or profession.</p>
    </div>
""", unsafe_allow_html=True)

# ======================
# RAG PIPELINE SETUP
# ======================
@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=os.path.join(os.getcwd(), "chroma_db"), 
        embedding_function=embeddings
    )
    return vectordb, embeddings

vectordb, embeddings = load_vectordb()

# Lazily load or create RAG components to speed up startup time
def get_rag_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=st.secrets["GOOGLE_API_KEY"], 
        temperature=0.2
    )

    
    prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert Nigerian legal assistant specializing in the Nigeria Data Protection Act (NDPA). "
        "Your task is to analyze the user's inquiry and provide professional, legally grounded insights based on the provided context.\n\n"
        
        "CRITICAL RULES:\n"
        "1. Maintain your persona as a helpful Nigerian legal expert at all times.\n"
        "2. Base your guidance strictly on the provided Context. Do not invent facts.\n"
        "3. If the provided context does not contain the exact answer but is related, summarize what the context *does* say about the topic to help the user.\n"
        "4. Answer only using the supplied context. If the answer is not contained in the context, say you do not know.'\n\n"
        
        "Context:\n{context}"
    )),
    ("human", "{input}")
])

    
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    question_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_chain)

# ======================
# SIDEBAR CONTROLS
# ======================
st.sidebar.title("⚙️ Controls")

if st.sidebar.button("Clear Conversation History"):
    st.session_state.messages = []

# ======================
# CHAT MEMORY
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



# ======================
# CHAT INPUT
# ======================
question = st.chat_input("Ask a legal question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documents..."):
            rag_chain = get_rag_chain()
            response = rag_chain.invoke({"input": question})
            answer = response["answer"]
            
            st.markdown(answer)
            
            if "context" in response:
                with st.expander("📄 Retrieved Sources"):
                    for i, doc in enumerate(response["context"], start=1):
                        st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i}</strong><br> 
                                {doc.page_content[:500]}...
                            </div>
                        """, unsafe_allow_html=True)
                        
            st.session_state.messages.append({"role": "assistant", "content": answer})
