"""Schema cho dữ liệu Gemini trả về và hàm chuyển sang format `Chapter` của frontend.

Gemini không hỗ trợ `dict[str, str]` trong response_schema nên options được mô tả
dưới dạng list 4 phần tử, sau đó chuyển thành {"1": ..., "4": ...} khi xuất ra.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class GeminiQuestion(BaseModel):
    question_number: int
    question_text: Optional[str] = Field(
        default=None,
        description="Nội dung câu hỏi. Giữ phần gạch chân bằng thẻ <u>...</u>. Với câu xếp ★ dùng '_____' cho chỗ trống và '★' cho ô sao.",
    )
    word: Optional[str] = Field(default=None, description="Chỉ dùng cho dạng hỏi nghĩa/cách dùng của một từ.")
    options: List[str] = Field(description="Các lựa chọn theo thứ tự, từ 2 đến 4 phần tử.")
    options_vi: List[str] = Field(
        default_factory=list,
        description="Nghĩa tiếng Việt từng lựa chọn (cùng số phần tử với options). Chỉ điền khi nội dung là tiếng nước ngoài; nếu là tiếng Việt thì để mảng rỗng.",
    )
    answer: int = Field(description="Số thứ tự (bắt đầu từ 1) của đáp án đúng.")
    answer_found_in_pdf: bool = Field(
        description="true nếu đáp án lấy từ bảng đáp án trong PDF; false nếu bạn tự suy luận.",
    )
    ordered_sequence: Optional[List[int]] = Field(
        default=None, description="Chỉ cho dạng xếp ★ (4 lựa chọn): thứ tự đúng của 4 lựa chọn."
    )
    full_sentence: Optional[str] = Field(default=None, description="Chỉ cho dạng xếp ★: câu hoàn chỉnh.")


class GeminiSection(BaseModel):
    section_name: str = Field(description="Ví dụ '問題 1'.")
    instruction: Optional[str] = Field(default=None, description="Câu hướng dẫn của phần thi.")
    questions: List[GeminiQuestion]


class GeminiExam(BaseModel):
    title: Optional[str] = Field(default=None, description="Tên của đề/bài kiểm tra theo tiêu đề trong PDF; để trống nếu không có.")
    sections: List[GeminiSection]


def to_chapter(exam: GeminiExam, chapter_id: str, chapter_name: str, subtitle: str) -> tuple[dict, List[str]]:
    """Chuyển sang format Chapter của frontend và trả về danh sách cảnh báo."""
    warnings: List[str] = []
    sections = []
    for sec in exam.sections:
        questions = []
        for q in sec.questions:
            tag = f"{sec.section_name} câu {q.question_number}"
            n_opts = len(q.options)
            if not 2 <= n_opts <= 4:
                warnings.append(f"{tag}: có {n_opts} lựa chọn (hỗ trợ 2-4), đã bỏ qua")
                continue
            if not 1 <= q.answer <= n_opts:
                warnings.append(f"{tag}: đáp án {q.answer} không hợp lệ, đã bỏ qua")
                continue
            if not q.answer_found_in_pdf:
                warnings.append(f"{tag}: đáp án do AI suy luận, cần kiểm tra")
            has_vi = len(q.options_vi) == n_opts
            if q.options_vi and not has_vi:
                warnings.append(f"{tag}: bản dịch không khớp số lựa chọn, bỏ qua options_vi")
            if q.ordered_sequence is not None and sorted(q.ordered_sequence) != list(range(1, n_opts + 1)):
                warnings.append(f"{tag}: ordered_sequence không phải hoán vị của các lựa chọn")
                q.ordered_sequence = None

            item: dict = {"question_number": q.question_number}
            if q.question_text:
                item["question_text"] = q.question_text
            if q.word:
                item["word"] = q.word
            item["options"] = {str(i + 1): t for i, t in enumerate(q.options)}
            item["answer"] = q.answer
            if has_vi:
                item["options_vi"] = {str(i + 1): t for i, t in enumerate(q.options_vi)}
            if q.ordered_sequence:
                item["ordered_sequence"] = q.ordered_sequence
            if q.full_sentence:
                item["full_sentence"] = q.full_sentence
            questions.append(item)

        if not questions:
            warnings.append(f"{sec.section_name}: không có câu hỏi hợp lệ, bỏ qua")
            continue
        section: dict = {"section_name": sec.section_name}
        if sec.instruction:
            section["instruction"] = sec.instruction
        section["questions"] = questions
        sections.append(section)

    chapter = {
        "chapter_id": chapter_id,
        "chapter_name": chapter_name,
        "category": "exam",
        "subtitle": subtitle,
        "sections": sections,
    }
    return chapter, warnings
