import os
import json
import logging
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.loader import load_and_split_documents
from src.vector_db import build_vector_store, load_vector_store
from src.rag_chain import create_rag_chain, RAGPipeline
from src.config import FAISS_INDEX_PATH, LLM_MODEL, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Biến global lưu trữ pipeline RAG
rag_pipeline: Optional[RAGPipeline] = None

def get_or_init_pipeline() -> RAGPipeline:
    global rag_pipeline
    if rag_pipeline is not None:
        return rag_pipeline

    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if not os.path.exists(index_file):
        logger.info("Chưa có index, đang xây dựng index từ data/...")
        chunks = load_and_split_documents()
        if not chunks:
            raise RuntimeError("Thư mục data/ không có tài liệu PDF nào để tạo index.")
        vectorstore = build_vector_store(chunks)
    else:
        logger.info("Đang tải FAISS vectorstore...")
        vectorstore = load_vector_store()

    rag_pipeline = create_rag_chain(vectorstore)
    return rag_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng (Lifespan event handlers thay thế cho on_event)."""
    try:
        get_or_init_pipeline()
        logger.info("RAG Pipeline đã được khởi tạo thành công.")
    except Exception as e:
        logger.warning(f"Chưa khởi tạo được RAG Pipeline khi khởi động: {e}")
    yield

app = FastAPI(
    title="JLPT RAG Chatbot API",
    description="API hỏi đáp tài liệu và đề thi tiếng Nhật JLPT N2 sử dụng FAISS và Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS để cho phép frontend gọi từ bất kỳ domain nào
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' hoặc 'assistant'")
    content: str = Field(..., description="Nội dung tin nhắn")

class ChatRequest(BaseModel):
    message: str = Field(..., description="Nội dung câu hỏi của người dùng")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Lịch sử hội thoại trước đó")

class SourceItem(BaseModel):
    file_name: str
    page: Optional[int] = None
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

def convert_history(history: List[ChatMessage]) -> List[BaseMessage]:
    """Chuyển đổi history dạng JSON sang danh sách BaseMessage của LangChain."""
    messages = []
    for msg in history:
        if msg.role.lower() in ["user", "human"]:
            messages.append(HumanMessage(content=msg.content))
        elif msg.role.lower() in ["assistant", "ai", "bot"]:
            messages.append(AIMessage(content=msg.content))
    return messages

def extract_source_items(docs: list) -> List[SourceItem]:
    sources = []
    for doc in docs:
        file_name = doc.metadata.get("file_name") or os.path.basename(doc.metadata.get("source", "Không rõ"))
        page = doc.metadata.get("page", None)
        page_val = page + 1 if page is not None else None
        snippet = doc.page_content.strip()[:200]
        sources.append(SourceItem(
            file_name=file_name,
            page=page_val,
            snippet=snippet
        ))
    return sources

# Endpoints
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "JLPT RAG Chatbot API",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "endpoints": {
            "chat_sync": "POST /api/chat",
            "chat_stream": "POST /api/chat/stream",
            "rebuild_index": "POST /api/index/rebuild",
            "ping": "GET /ping"
        }
    }

@app.get("/ping")
@app.get("/health")
def ping():
    index_exists = os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss"))
    return {
        "status": "healthy",
        "index_ready": index_exists
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """API chat trả về toàn bộ câu trả lời kèm danh sách nguồn trích dẫn."""
    try:
        pipeline = get_or_init_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể khởi tạo RAG: {str(e)}")

    chat_history = convert_history(request.history or [])
    try:
        docs = await pipeline.aget_relevant_documents(request.message, chat_history)
        result = await pipeline.qa_chain.ainvoke({
            "context": docs,
            "question": request.message,
            "chat_history": chat_history
        })
        sources = extract_source_items(docs)
        return ChatResponse(answer=result, sources=sources)
    except Exception as e:
        logger.error(f"Lỗi khi xử lý chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """API chat streaming câu trả lời bằng Server-Sent Events (SSE)."""
    try:
        pipeline = get_or_init_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể khởi tạo RAG: {str(e)}")

    chat_history = convert_history(request.history or [])

    async def event_generator():
        try:
            # 1. Truy xuất tài liệu ngữ cảnh
            docs = await pipeline.aget_relevant_documents(request.message, chat_history)
            sources = extract_source_items(docs)
            sources_payload = [s.model_dump() for s in sources]
            yield f"event: sources\ndata: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

            # 2. Stream từng token từ LLM
            async for token in pipeline.qa_chain.astream({
                "context": docs,
                "question": request.message,
                "chat_history": chat_history
            }):
                yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            # 3. Kết thúc stream
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.error(f"Lỗi khi streaming: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.post("/api/index/rebuild")
def rebuild_index():
    """Quét lại toàn bộ tài liệu trong data/ và tạo lại FAISS index mới."""
    global rag_pipeline
    import shutil
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        chunks = load_and_split_documents()
        if not chunks:
            raise HTTPException(status_code=400, detail="Không tìm thấy file PDF nào trong data/")
        vectorstore = build_vector_store(chunks)
        rag_pipeline = create_rag_chain(vectorstore)
        return {
            "status": "success",
            "message": f"Đã xây dựng lại FAISS index thành công với {len(chunks)} chunks."
        }
    except Exception as e:
        logger.error(f"Lỗi khi rebuild index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
