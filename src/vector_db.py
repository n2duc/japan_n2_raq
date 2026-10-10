import os
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from src.config import GOOGLE_API_KEY, EMBEDDING_MODEL, FAISS_INDEX_PATH

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY chưa được cấu hình trong file .env")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

def build_vector_store(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

def load_vector_store() -> FAISS:
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore