import os
import re
import json
import logging
from typing import Any, Dict, List, Optional

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

# Một chunk câu hỏi JLPT là đơn vị "nguyên tử": không cắt nhỏ. Chỉ cắt khi quá dài bất thường.
MAX_ATOMIC_CHARS = 2000

_BLANK_RE = re.compile(r"[（(]\s*[）)]")      # （ ）
_STAR_SLOT_RE = re.compile(r"_*★_*|_+")        # ___ / ★ / __★__
_PASSAGE_MARK_RE = re.compile(r"\[\s*(\d+)\s*\]")  # [ 1 ]


# --------------------------------------------------------------------------- #
# JSON (đề thi / chương học JLPT)
# --------------------------------------------------------------------------- #
def _classify_section(section: Dict[str, Any]) -> str:
    """Phân loại dạng bài để dùng làm metadata filter khi retrieval."""
    instr = section.get("instruction", "")
    if section.get("context"):
        return "passage_grammar"
    if "★" in instr:
        return "sentence_order"
    if "読み方" in instr:
        return "reading"
    if "漢字" in instr:
        return "kanji_writing"
    if "意味が最も近い" in instr:
        return "synonym"
    if "使い方" in instr:
        return "usage"
    if "次の文の" in instr:
        return "grammar_fill"
    return "vocab_fill"


def _fill_blank(text: str, answer_text: str) -> str:
    """Điền đáp án đúng vào chỗ trống （ ） để có câu hoàn chỉnh."""
    return _BLANK_RE.sub(answer_text, text, count=1) if _BLANK_RE.search(text) else text


def _rebuild_ordered_sentence(text: str, options: Dict[str, str], seq: List[int]) -> Optional[str]:
    """Dạng ★: ghép lại câu đúng từ ordered_sequence."""
    slots = list(_STAR_SLOT_RE.finditer(text))
    if len(slots) != len(seq):
        return None
    out, last = [], 0
    for m, idx in zip(slots, seq):
        out.append(text[last:m.start()])
        out.append(options[str(idx)])
        last = m.end()
    out.append(text[last:])
    return re.sub(r"\s+", "", "".join(out))


def _format_options(q: Dict[str, Any]) -> str:
    opts, vi = q.get("options", {}), q.get("options_vi", {})
    lines = []
    for k in sorted(opts, key=lambda x: int(x)):
        line = f"{k}. {opts[k]}"
        if vi.get(k):
            line += f" — {vi[k]}"
        lines.append(line)
    return "\n".join(lines)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？\n])", text)
    return [p for p in parts if p.strip()]


def _local_context(passage: str, marker_no: int, window: int = 1) -> str:
    """Lấy câu chứa [ n ] cùng `window` câu trước/sau (thay vì nhồi cả đoạn văn)."""
    sents = _split_sentences(passage)
    for i, s in enumerate(sents):
        if any(int(n) == marker_no for n in _PASSAGE_MARK_RE.findall(s)):
            return "".join(sents[max(0, i - window): i + window + 1]).strip()
    return passage


def _make_doc(content: str, base_meta: Dict[str, Any], **extra) -> Document:
    meta = {**base_meta, **extra}
    return Document(page_content=content.strip(), metadata=meta)


def _chunks_from_exam_json(data: Dict[str, Any], source: str) -> List[Document]:
    exam_id = str(data.get("chapter_id", os.path.splitext(os.path.basename(source))[0]))
    exam_name = data.get("chapter_name", exam_id)
    category = data.get("category", "chapter")
    header = f"[JLPT N2 | {exam_name}]"

    docs: List[Document] = []
    for s_idx, section in enumerate(data.get("sections", []), start=1):
        s_name = section.get("section_name", f"section_{s_idx}")
        s_type = _classify_section(section)
        instruction = section.get("instruction", "")
        passage = section.get("context")

        base = {
            "source": source,
            "file_name": os.path.basename(source),
            "exam_id": exam_id,
            "exam_name": exam_name,
            "category": category,
            "section_name": s_name,
            "section_type": s_type,
        }

        # --- Đoạn văn (問題3 文章の文法): 1 chunk riêng cho cả đoạn ---
        if passage:
            docs.append(_make_doc(
                f"{header} {s_name}\n" + (f"Hướng dẫn: {instruction}\n" if instruction else "") + f"Đoạn văn:\n{passage}",
                base, chunk_type="passage", chunk_id=f"{exam_id}:{s_idx}:passage",
            ))

        for q_idx, q in enumerate(section.get("questions", []), start=1):
            opts = q.get("options", {})
            ans = str(q.get("answer", ""))
            ans_text = opts.get(ans, "")
            ans_vi = q.get("options_vi", {}).get(ans, "")

            # Số thứ tự: với đoạn văn, số ô trống = vị trí trong list (file mẫu có số bị trùng/sai)
            declared_no = q.get("question_number")
            q_no = q_idx if passage else declared_no
            if passage and declared_no != q_idx:
                logger.warning("%s/%s: question_number=%s lệch vị trí %s → dùng %s",
                               exam_id, s_name, declared_no, q_idx, q_idx)

            # Nội dung đề
            if passage:
                q_text = _local_context(passage, q_no)
                question_line = f"Ngữ cảnh ô trống [ {q_no} ]: {q_text}"
                full_sentence = None
            elif "word" in q:
                question_line = f"Từ cần dùng: {q['word']}"
                full_sentence = None
            else:
                q_text = q.get("question_text", "")
                question_line = f"Đề: {q_text}"
                if q.get("ordered_sequence"):
                    full_sentence = _rebuild_ordered_sentence(q_text, opts, q["ordered_sequence"])
                else:
                    full_sentence = _fill_blank(q_text, ans_text) if s_type in (
                        "grammar_fill", "vocab_fill") else None

            parts = [
                f"{header} {s_name} — Câu {q_no}",
                f"Dạng bài: {s_type}" + (f" | Hướng dẫn: {instruction}" if instruction else ""),
                question_line,
                "Lựa chọn:\n" + _format_options(q),
                f"Đáp án đúng: {ans}. {ans_text}" + (f" — {ans_vi}" if ans_vi else ""),
            ]
            if full_sentence:
                parts.append(f"Câu hoàn chỉnh: {full_sentence}")
            if q.get("ordered_sequence"):
                order = " → ".join(opts[str(i)] for i in q["ordered_sequence"])
                parts.append(f"Thứ tự đúng: {order} (đáp án ★ là vị trí số {ans})")

            docs.append(_make_doc(
                "\n".join(parts), base,
                chunk_type="question",
                chunk_id=f"{exam_id}:{s_idx}:{q_idx}",
                question_number=q_no,
                answer=ans,
                answer_text=ans_text,
                has_passage=bool(passage),
            ))
    return docs


def _load_json_chunks(data_path: str) -> List[Document]:
    docs: List[Document] = []
    for root, _, files in os.walk(data_path):
        for fn in sorted(files):
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(root, fn)
            try:
                # utf-8-sig để chịu được file có BOM
                with open(path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "sections" in data:
                    docs.extend(_chunks_from_exam_json(data, path))
                else:
                    logger.warning("Bỏ qua %s: không đúng schema đề thi (thiếu 'sections')", path)
            except Exception as e:  # noqa: BLE001
                logger.error("Lỗi đọc %s: %s", path, e)
    return docs


# --------------------------------------------------------------------------- #
# PDF (giữ nguyên hướng cũ, sửa nhẹ separator tiếng Nhật)
# --------------------------------------------------------------------------- #
def _load_pdf_chunks(data_path: str) -> List[Document]:
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=DEFAULT_LOADER)
    documents = loader.load()
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        strip_whitespace=True,
        keep_separator="end",  # giữ 。！？ ở cuối câu thay vì đẩy xuống đầu chunk sau
        separators=["\n\n", "\n", "。", "！", "？", "；", "、", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for c in chunks:
        if "source" in c.metadata:
            c.metadata["file_name"] = os.path.basename(c.metadata["source"])
        c.metadata.setdefault("chunk_type", "pdf_text")
    return chunks


# --------------------------------------------------------------------------- #
def load_and_split_documents(data_path: str = "./data") -> List[Document]:
    """Nạp dữ liệu JLPT.

    - JSON đề thi/chương: chunk theo CÂU HỎI (1 câu = 1 chunk, kèm đáp án + giải nghĩa VI),
      đoạn văn 問題3 có chunk riêng; không dùng text splitter để tránh cắt đôi câu hỏi/đáp án.
    - PDF: RecursiveCharacterTextSplitter với separator tối ưu cho tiếng Nhật.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        return []

    chunks = _load_json_chunks(data_path)

    # Phòng hờ chunk quá dài (hiếm): cắt bằng splitter, vẫn giữ metadata
    safe: List[Document] = []
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=MAX_ATOMIC_CHARS, chunk_overlap=CHUNK_OVERLAP,
        keep_separator="end", separators=["\n\n", "\n", "。", " ", ""],
    )
    for d in chunks:
        safe.extend(fallback.split_documents([d]) if len(d.page_content) > MAX_ATOMIC_CHARS else [d])

    try:
        safe.extend(_load_pdf_chunks(data_path))
    except Exception as e:  # noqa: BLE001
        logger.error("Lỗi đọc PDF: %s", e)

    return safe