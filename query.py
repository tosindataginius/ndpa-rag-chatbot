# query.py
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# 1. Initialize the exact same local embedding model used for indexing
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load the existing local database
print("Loading vector database...")
vectordb = Chroma(
    persist_directory= os.path.join(os.getcwd(), "chroma_db"),
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5}) # Retrieves top 5 matching blocks

# retriever = vectordb.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k":5,
#         "fetch_k":20
#     }
# )

# 3. Initialize Google Gemini (using the free tier)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# 4. Create the System Prompt Architecture
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


# 5. Build and run the RAG Pipeline
question_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_chain)

# Test Question
#query = "What are the main objectives of the Nigeria Data Protection Act 2023?"
query = "Summarize the key provisions of the Nigeria Data Protection Act 2023 and quote the source?" # Try this one for a more complex question!

print(f"\nAsking: {query}\nThinking...")

response = rag_chain.invoke({"input": query})

print("\n--- Answer ---")
print(response["answer"])
