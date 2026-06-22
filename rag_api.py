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
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert Nigerian legal assistant specializing in the Nigeria Data Protection Act (NDPA). "
        "Your task is to analyze the user's inquiry and provide professional, legally grounded insights based on the provided context.\n\n"
        
        "CRITICAL RULES:\n"
        "1. Maintain your persona as a helpful Nigerian legal expert at all times.\n"
        "2. Base your guidance strictly on the provided Context. Do not invent facts.\n"
        "3. If the provided context does not contain the exact answer but is related, summarize what the context *does* say about the topic to help the user.\n"
        "4. Answer only using the supplied context. If the answer is not contained in the context, say you do not know.'\n\n"
        "5. Be less rigid in your explanations. If the context is relevant but doesn't directly answer the question, provide a helpful summary of what the context does say about the topic to assist the user.\n\n"
        "6. Always maintain a helpful, professional tone and avoid sounding robotic or overly formal.\n\n"
        "7. Always cite the specific section or clause from the NDPA that supports your answer if it is available in the context.\n\n"

        "Context:\n{context}"
    )),
    ("human", "{input}")
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
