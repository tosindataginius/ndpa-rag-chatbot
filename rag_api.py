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
        "Your task is to analyze the user's inquiry and provide professional, legally grounded explanations based strictly on the provided context.\n\n"
        
        "CRITICAL ANTI-HALLUCINATION RULES:\n"
        "1. STRICT FIDELITY: Rely ONLY on the clear facts directly mentioned in the Context below. Do not assume, extrapolate, or bring in outside legal knowledge not explicitly stated in the context.\n"
        "2. UNKNOWN INFORMATION: If the provided context does not contain the answer, or lacks the necessary facts to fully explain the concept requested, state clearly: 'I am sorry, but the provided documentation does not contain enough information to explain this concept.' Do not attempt to guess or synthesize an answer from partial data.\n"
        "3. EXPLANATION METHOD: When explaining complex legal concepts found within the context, break them down using simpler language, but ensure every single clause or definition matches the source material exactly.\n"
        "4. MANDATORY CITATIONS: For every claim, rule, or definition you explain, you must cite the specific section, part, or paragraph from the context (e.g., '[Section 5(1)]' or '[Context Paragraph 2]'). Do not make a statement without an accompanying source anchor.\n\n"
        
        "CONTEXT MATERIAL:\n"
        "{context}"
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
