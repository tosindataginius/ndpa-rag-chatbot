# ingest.py
import os
import warnings
   
# Suppress the LangChain community sunset warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Core LangChain Modules
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings



# Absolute file path to your document
pdf_path = r"C:\Users\USER\Desktop\tensorvirtual\rag-gemini\data\Nigeria_Data_Protection_Act2023.pdf"

# 1. Load the PDF
print("Loading PDF document...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split the document into chunks
print("Splitting text into manageable chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
)
chunks = splitter.split_documents(documents)

# Load a model with a large context window (8,192 tokens)
#embeddings = model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


print("Initializing open-source embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory= r"C:\Users\USER\Desktop\tensorvirtual\rag-gemini\chroma_db"
)

#vectordb.persist()

print("PDF successfully indexed.")