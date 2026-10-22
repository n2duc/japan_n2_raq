"""Rate limit đơn giản trong bộ nhớ (một process). Đổi sang Redis nếu chạy nhiều worker."""
import os
import time
from collections import defaultdict, deque
from fastapi import HTTPException, Request

HOURLY = int(os.getenv("EXAM_IMPORT_PER_HOUR", "3"))
DAILY = int(os.getenv("EXAM_IMPORT_PER_DAY", "10"))
MAX_PDF_BYTES = int(os.getenv("EXAM_IMPORT_MAX_MB", "20")) * 1024 * 1024

_hits: dict[str, deque] = defaultdict(deque)


def client_key(request: Request) -> str:
    device = request.headers.get("x-device-id", "")[:64]
    ip = request.client.host if request.client else "unknown"
    return f"{ip}|{device}"


def check_rate_limit(request: Request) -> None:
    key = client_key(request)
    now = time.time()
    q = _hits[key]
    while q and now - q[0] > 86400:
        q.popleft()
    if len(q) >= DAILY or sum(1 for t in q if now - t <= 3600) >= HOURLY:
        raise HTTPException(status_code=429, detail="Đã vượt giới hạn số lần import, vui lòng thử lại sau")
    q.append(now)


def validate_pdf(data: bytes, label: str) -> None:
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail=f"{label} không phải file PDF hợp lệ")
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} vượt quá {MAX_PDF_BYTES // 1024 // 1024}MB")
