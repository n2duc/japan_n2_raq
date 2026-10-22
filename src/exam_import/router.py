import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from .limits import MAX_PDF_BYTES, check_rate_limit, validate_pdf
from .service import ExamImportError, extract_exam

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/exams", tags=["exam-import"])


async def read_limited(file: UploadFile, label: str) -> bytes:
    data = await file.read(MAX_PDF_BYTES + 1)
    validate_pdf(data, label)
    return data


@router.post("/import")
async def import_exam(
    request: Request,
    exam_pdf: UploadFile = File(...),
    answer_pdf: Optional[UploadFile] = File(default=None),
    title: Optional[str] = Form(default=None, max_length=120),
):
    """Nhận PDF đề bất kỳ (và PDF đáp án nếu có), trả về JSON đề theo format Chapter.

    `title` tùy chọn; không có thì lấy tên từ nội dung PDF.

    Stateless: không lưu PDF hay kết quả trên server, client tự lưu.
    """
    exam_bytes = await read_limited(exam_pdf, "File đề")
    answer_bytes = await read_limited(answer_pdf, "File đáp án") if answer_pdf else None
    check_rate_limit(request)

    cid = f"custom-{uuid.uuid4().hex[:8]}"
    try:
        chapter, warnings = await extract_exam(
            exam_bytes, answer_bytes, cid, title, "PDF"
        )
    except ExamImportError as e:
        raise HTTPException(status_code=502, detail=str(e))

    logger.info("exam import ok: %s sections, %s warnings", len(chapter["sections"]), len(warnings))
    return {"chapter": chapter, "warnings": warnings}
