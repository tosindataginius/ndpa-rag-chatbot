# main.py
import os
import warnings
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LangChain Modules
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="Nigeria Data Protection Act 2023 RAG API",
    description="An API to query legal documents using local vector search and Gemini 2.5 Flash.",
    version="1.0"
)

# Define the expected JSON request body schema
class QueryRequest(BaseModel):
    question: str

# 1. Global Setup (Loads once when the server starts up)
print("Initializing local HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Connecting to local Chroma DB...")
DB_DIR = os.path.join(os.getcwd(), "chroma_db")
if not os.path.exists(DB_DIR):
    raise FileNotFoundError(f"Chroma directory not found at {DB_DIR}. Please run ingest.py first!")


# Load the existing local database
vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

print("Initializing Google Gemini API...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# 2. Build Prompt & Chains
system_prompt = (
    "You are an expert assistant specialized in Nigerian legal documents.\n"
    "Answer the user's question using strictly the retrieved context below. "
    "If you do not know the answer based on the context, say that you do not know.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_chain)
print("RAG API Pipeline ready to accept connections.")

# Health check route
@app.get("/")
async def root():
    return {"status": "online", "message": "Legal RAG Server is running."}


# 3. Create the API Endpoint  
@app.post("/query")
async def ask_legal_agent(request: QueryRequest):
    """
    Accepts a question via POST JSON body, queries the local DB, 
    and returns the context-aware response from Gemini.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question field cannot be empty.")
        
    try:
        # Invoke the LangChain retrieval chain
        response = rag_chain.invoke({"input": request.question})
        
        return {
            "question": request.question,
            "answer": response["answer"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Health check route
@app.get("/")
async def root():
    return {"status": "online", "message": "Legal RAG Server is running."}
