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
        "Your task is to analyze the user's inquiry and provide professional, legally grounded explanations based strictly on the provided context.\n\n"
        
        "CRITICAL ANTI-HALLUCINATION RULES:\n"
        "1. STRICT FIDELITY: Rely ONLY on the clear facts directly mentioned in the Context below. Do not assume, extrapolate, or bring in outside legal knowledge not explicitly stated in the context.\n"
        "2. UNKNOWN INFORMATION: If the provided context does not contain the answer, or lacks the necessary facts to fully explain the concept requested, state clearly: 'I am sorry, but the provided documentation does not contain enough information to explain this concept.' Do not attempt to guess or synthesize an answer from partial data.\n"
        "3. EXPLANATION METHOD: When explaining complex legal concepts found within the context, break them down using simpler language, but ensure every single clause or definition matches the source material exactly.\n"
        "4. MANDATORY CITATIONS: For every claim, rule, or definition you explain, you must cite the specific section, part, or paragraph from the context (e.g., '[Section 5(1)]'). Do not make a statement without an accompanying source anchor.\n\n"
        
        "CONTEXT MATERIAL:\n"
        "{context}"
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



with st.sidebar:
    st.markdown("Developed by Oluwasegun Oluwatosin (tosindataginius)")
    st.link_button("Visit my LinkedIn Profile", "https://www.linkedin.com/in/oluwatosin-oluwasegun-1a9266288/")
    st.link_button("Visit my GitHub Profile", "https://github.com/tosindataginius")
