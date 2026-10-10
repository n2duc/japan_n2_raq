import os
import sys
import shutil
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Bắt buộc terminal sử dụng mã hóa UTF-8 với cơ chế thay thế ký tự lỗi
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.loader import load_and_split_documents
from src.vector_db import build_vector_store, load_vector_store
from src.rag_chain import create_rag_chain
from src.config import FAISS_INDEX_PATH

def safe_input(prompt: str = "") -> str:
    """A wrapper function to safely handle input and prevent crashes when the Vietnamese input method sends invalid bytes."""
    try:
        return input(prompt).strip()
    except UnicodeDecodeError:
        raw_bytes = sys.stdin.buffer.readline()
        return raw_bytes.decode("utf-8", errors="ignore").strip()

def print_help():
    print("""
Các lệnh hữu ích:
  - 'exit', 'quit', 'q': Thoát chương trình
  - '/clear': Xóa lịch sử cuộc trò chuyện hiện tại
  - '/rebuild': Quét lại thư mục data/ và tạo lại FAISS index
  - '/help': Xem hướng dẫn
    """)

def display_sources(docs: list):
    """Display a summary of sources and reference documents."""
    if not docs:
        return
    print("📚 Nguồn tham chiếu:")
    seen_sources = set()
    for doc in docs:
        source = doc.metadata.get("file_name") or os.path.basename(doc.metadata.get("source", "Không rõ"))
        page = doc.metadata.get("page", None)
        key = (source, page)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        page_str = f" - Trang {page + 1}" if page is not None else ""
        print(f"  [{len(seen_sources)}] {source}{page_str}")
    print()

def main():
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")

    if not os.path.exists(index_file):
        print("Chưa có FAISS index. Đang đọc tài liệu từ data/...")
        chunks = load_and_split_documents()
        if not chunks:
            print("Không tìm thấy file PDF nào trong data/. Hãy copy file PDF vào thư mục data/")
            sys.exit(1)

        print(f"Đã đọc và chia thành {len(chunks)} chunks. Đang tạo embeddings vào FAISS...")
        vectorstore = build_vector_store(chunks)
        print("Tạo FAISS index thành công!")
    else:
        print("Đang tải FAISS index có sẵn...")
        vectorstore = load_vector_store()

    chain = create_rag_chain(vectorstore)
    chat_history: List[BaseMessage] = []

    print("\n" + "="*55)
    print("🤖 Chatbot RAG Tiếng Nhật JLPT (Streaming Enabled)")
    print("Gõ câu hỏi của bạn, hoặc gõ '/help' để xem các lệnh.")
    print("="*55 + "\n")

    while True:
        try:
            query = safe_input("Bạn: ")
        except (KeyboardInterrupt, EOFError):
            print("\nĐang thoát chương trình...")
            break

        if not query:
            continue

        cmd = query.lower()
        if cmd in ["exit", "quit", "q"]:
            print("Tạm biệt!")
            break

        if cmd == "/help":
            print_help()
            continue

        if cmd == "/clear":
            chat_history.clear()
            print("🧹 Đã làm mới lịch sử cuộc trò chuyện.\n")
            continue

        if cmd == "/rebuild":
            print("Đang xóa index cũ và tạo lại...")
            if os.path.exists(FAISS_INDEX_PATH):
                shutil.rmtree(FAISS_INDEX_PATH)
            chunks = load_and_split_documents()
            if not chunks:
                print("Không tìm thấy file nào trong data/.")
                continue
            print(f"Đã chia thành {len(chunks)} chunks mới. Đang build vector store...")
            vectorstore = build_vector_store(chunks)
            chain = create_rag_chain(vectorstore)
            chat_history.clear()
            print("✅ Đã rebuild FAISS index thành công!\n")
            continue

        try:
            print("\n🤖 Bot đang suy nghĩ...", end="\r", flush=True)
            # 1. Truy xuất tài liệu ngữ cảnh
            docs = chain.get_relevant_documents(query, chat_history)

            # 2. Xóa thông báo loading và Stream câu trả lời
            print(" " * 30, end="\r")
            print("Bot: ", end="", flush=True)
            full_chunks = []
            for chunk in chain.qa_chain.stream({
                "context": docs,
                "question": query,
                "chat_history": chat_history
            }):
                print(chunk, end="", flush=True)
                full_chunks.append(chunk)
            print("\n")

            answer = "".join(full_chunks)

            # 3. Hiển thị nguồn trích dẫn
            display_sources(docs)

            # 4. Lưu lại lịch sử trò chuyện
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=answer))
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

        except Exception as e:
            print(f"\n❌ Đã xảy ra lỗi khi xử lý câu hỏi: {e}\n")

if __name__ == "__main__":
    main()