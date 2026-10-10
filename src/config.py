import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("GOOGLE_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./vectorstore/faiss_index")

# Retriever parameters
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.2"))

# Document chunking parameters (optimized for Japanese and JLPT materials)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))