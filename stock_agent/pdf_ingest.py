from __future__ import annotations

from pathlib import Path
import hashlib
from datetime import datetime, date
import re

from db import source_doc_exists, insert_source_document, insert_source_document_chunks


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def extract_pdf_text(path: Path) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    return " ".join(page.get_text("text") for page in doc)


def chunk_text(text: str, max_chars: int = 1400) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars) if text[i:i+max_chars].strip()]


def infer_doc_date_from_filename(name: str) -> date | None:
    # 1) YYYY-MM-DD
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", name)
    if m:
        y, mo, d = map(int, m.groups())
        return date(y, mo, d)

    # 2) YYYYMMDD
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if m:
        y, mo, d = map(int, m.groups())
        return date(y, mo, d)

    # 3) Q[1-4]_YYYY
    m = re.search(r"\bQ([1-4])[_\- ]?(20\d{2})\b", name, re.IGNORECASE)
    if m:
        q = int(m.group(1))
        y = int(m.group(2))
        return {1: date(y,3,31), 2: date(y,6,30), 3: date(y,9,30), 4: date(y,12,31)}[q]

    return None


def ingest_pdfs_for_entity(conn, pdf_root: str, entity_id: str, entity_type: str = "public") -> int:
    root = Path(pdf_root) / entity_id
    if not root.exists():
        return 0

    inserted = 0
    for pdf_path in root.rglob("*.pdf"):
        file_hash = sha256_file(pdf_path)
        if source_doc_exists(conn, file_hash, entity_id):
            continue

        file_mtime = datetime.fromtimestamp(pdf_path.stat().st_mtime)
        doc_date = infer_doc_date_from_filename(pdf_path.name) or file_mtime.date()

        text = extract_pdf_text(pdf_path)
        doc_id = file_hash  # stable identifier

        insert_source_document(
            conn,
            doc_id=doc_id,
            entity_id=entity_id,
            entity_type=entity_type,
            title=pdf_path.name,
            file_path=str(pdf_path),
            file_mtime=file_mtime,
            file_hash=file_hash,
            doc_date=doc_date,
            extracted_text=text,
        )

        chunks = chunk_text(text)
        insert_source_document_chunks(conn, doc_id=doc_id, entity_id=entity_id, chunks=chunks)

        inserted += 1

    return inserted
