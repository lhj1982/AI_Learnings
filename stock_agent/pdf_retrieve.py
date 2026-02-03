from __future__ import annotations

import re
from db import company_has_docs, get_recent_doc_ids, get_chunks_for_docs


def retrieve_pdf_chunks(conn, entity_id: str, query: str, max_chunks: int = 6, newest_docs: int = 3) -> list[str]:
    if not company_has_docs(conn, entity_id):
        return []

    doc_ids = get_recent_doc_ids(conn, entity_id, limit=newest_docs)
    chunks = get_chunks_for_docs(conn, entity_id, doc_ids)

    keywords = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 3}
    if not keywords:
        return chunks[:max_chunks]

    scored: list[tuple[int, str]] = []
    for c in chunks:
        words = set(re.findall(r"[A-Za-z0-9]+", c.lower()))
        score = len(words & keywords)
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_chunks]]
