import os
from typing import List, Dict, Any, Optional, AsyncIterator, Iterator
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GOOGLE_API_KEY, LLM_MODEL, RETRIEVER_K

RAG_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia giải đáp tài liệu và hỗ trợ ôn luyện tiếng Nhật (JLPT N2).
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA VÀO các đoạn ngữ cảnh (Context) được cung cấp bên dưới.

Yêu cầu thực hiện:
1. Dựa trên thông tin trong ngữ cảnh để trả lời. Nếu câu hỏi liên quan đến bài tập/đề thi tiếng Nhật, hãy giải thích rõ: nghĩa của câu, cấu trúc ngữ pháp, từ vựng hoặc lý do chọn/loại trừ đáp án.
2. Nếu thông tin không có trong ngữ cảnh, hãy trả lời trung thực: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp." Không tự bịa thông tin.
3. Trả lời rõ ràng, mạch lạc, súc tích và chính xác theo tài liệu gốc.

Ngữ cảnh tài liệu:
{context}
"""

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Cho trước lịch sử trò chuyện và câu hỏi mới nhất của người dùng.
Nếu câu hỏi có liên quan hoặc tham chiếu đến các lượt nói trước đó (ví dụ: "câu đó", "tại sao lại như vậy", "còn phần 2 thì sao?"),
hãy diễn đạt lại thành một câu hỏi độc lập (standalone question) để có thể tìm kiếm tài liệu chính xác mà không cần lịch sử.
KHÔNG trả lời câu hỏi, chỉ viết lại câu hỏi nếu cần, nếu câu hỏi đã độc lập thì giữ nguyên."""

def format_docs(docs: List[Document]) -> str:
    """Concatenate the content of the documents, including metadata (source, page number if available)."""
    if not docs:
        return "Không tìm thấy đoạn trích phù hợp."
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("file_name") or os.path.basename(doc.metadata.get("source", "Không rõ"))
        page = doc.metadata.get("page", None)
        page_info = f" (Trang {page + 1})" if page is not None else ""
        header = f"[Tài liệu {i} - Nguồn: {source}{page_info}]"
        formatted_chunks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(formatted_chunks)

class RAGPipeline:
    def __init__(self, vectorstore, k: int = RETRIEVER_K):
        self.vectorstore = vectorstore
        self.k = k
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY
        )

        # Contextualize question prompt & chain
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        self.contextualize_chain = contextualize_prompt | self.llm | StrOutputParser()

        # QA prompt & chain
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        self.qa_chain = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def resolve_question(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Nếu có lịch sử trò chuyện, viết lại thành câu hỏi độc lập để search vector."""
        if chat_history and len(chat_history) > 0:
            try:
                res = self.contextualize_chain.invoke({
                    "chat_history": chat_history,
                    "question": question
                })
                # Bắt buộc chuyển sang primitive str để Google GenAI SDK serialize đúng
                return str(res).strip() if res else str(question).strip()
            except Exception:
                return str(question).strip()
        return str(question).strip()

    async def aresolve_question(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Async viết lại câu hỏi độc lập."""
        if chat_history and len(chat_history) > 0:
            try:
                res = await self.contextualize_chain.ainvoke({
                    "chat_history": chat_history,
                    "question": question
                })
                # Bắt buộc chuyển sang primitive str để Google GenAI SDK serialize đúng
                return str(res).strip() if res else str(question).strip()
            except Exception:
                return str(question).strip()
        return str(question).strip()

    def get_relevant_documents(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> List[Document]:
        """Truy xuất các tài liệu phù hợp từ vector store."""
        search_query = str(self.resolve_question(question, chat_history)).strip()
        try:
            return self.retriever.invoke(search_query)
        except Exception:
            # Fallback với câu hỏi gốc dạng primitive str
            return self.retriever.invoke(str(question).strip())

    async def aget_relevant_documents(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> List[Document]:
        """Async truy xuất tài liệu từ vector store."""
        search_query = str(await self.aresolve_question(question, chat_history)).strip()
        try:
            return await self.retriever.ainvoke(search_query)
        except Exception:
            # Fallback với câu hỏi gốc dạng primitive str
            return await self.retriever.ainvoke(str(question).strip())

    def invoke(self, input_data: Any) -> Dict[str, Any]:
        """Execute synchronous RAG (returns the complete result)."""
        if isinstance(input_data, str):
            question = input_data
            chat_history = []
        elif isinstance(input_data, dict):
            question = input_data.get("question", "")
            chat_history = input_data.get("chat_history", [])
        else:
            raise ValueError("Input data must be str or dict")

        docs = self.get_relevant_documents(question, chat_history)
        answer = self.qa_chain.invoke({
            "context": docs,
            "question": question,
            "chat_history": chat_history
        })
        return {"answer": answer, "context": docs}

    def stream(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> Iterator[str]:
        """Stream response tokens one by one (synchronously for terminal/console)."""
        chat_history = chat_history or []
        docs = self.get_relevant_documents(question, chat_history)
        for chunk in self.qa_chain.stream({
            "context": docs,
            "question": question,
            "chat_history": chat_history
        }):
            yield chunk

    async def astream(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> AsyncIterator[str]:
        """Stream the response token by token (Async for Web API)."""
        chat_history = chat_history or []
        docs = await self.aget_relevant_documents(question, chat_history)
        async for chunk in self.qa_chain.astream({
            "context": docs,
            "question": question,
            "chat_history": chat_history
        }):
            yield chunk

def create_rag_chain(vectorstore, k: int = RETRIEVER_K) -> RAGPipeline:
    return RAGPipeline(vectorstore, k=k)