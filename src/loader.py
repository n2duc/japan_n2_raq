import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    DEFAULT_LOADER = PyMuPDFLoader
except ImportError:
    DEFAULT_LOADER = PyPDFLoader

logger = logging.getLogger(__name__)

def load_and_split_documents(data_path: str = "./data") -> List[Document]:
    """Read PDF documents (prioritizing PyMuPDF for its strong Japanese language support) and perform optimal chunking."""
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        return []

    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=DEFAULT_LOADER,
        show_progress=True
    )
    documents = loader.load()
    if not documents:
        return []

    # Optimize text splitter for Japanese (JLPT):
    # Prioritize splitting by paragraphs, lines, and Japanese punctuation marks (。, ！, ？)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Standardize metadata (shorten source file paths for readability)
    for chunk in chunks:
        if "source" in chunk.metadata:
            chunk.metadata["file_name"] = os.path.basename(chunk.metadata["source"])

    return chunks