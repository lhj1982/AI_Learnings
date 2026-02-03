from __future__ import annotations

import json
import os
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "stocks"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


# ---------------------------
# Daily analysis
# ---------------------------

def fetch_existing_daily(conn, day: str, entity_id: str) -> dict | None:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM stock_daily_analysis
            WHERE date = %s AND entity_id = %s
            """,
            (day, entity_id),
        )
        return cur.fetchone()


def upsert_daily_analysis(
    conn,
    day: str,
    entity_id: str,
    entity_type: str,
    fields: dict[str, Any],
    raw_context: dict[str, Any],
):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stock_daily_analysis
            (
                date, entity_id, entity_type,
                company_overview, recent_financials, valuation_metrics,
                technical_analysis, catalysts, risks, recommendation,
                raw_context
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (date, entity_id)
            DO UPDATE SET
                company_overview = EXCLUDED.company_overview,
                recent_financials = EXCLUDED.recent_financials,
                valuation_metrics = EXCLUDED.valuation_metrics,
                technical_analysis = EXCLUDED.technical_analysis,
                catalysts = EXCLUDED.catalysts,
                risks = EXCLUDED.risks,
                recommendation = EXCLUDED.recommendation,
                raw_context = EXCLUDED.raw_context
            """,
            (
                day,
                entity_id,
                entity_type,
                fields.get("company_overview"),
                fields.get("recent_financials"),
                fields.get("valuation_metrics"),
                fields.get("technical_analysis"),
                fields.get("catalysts"),
                fields.get("risks"),
                fields.get("recommendation"),
                json.dumps(raw_context),
            ),
        )
    conn.commit()


# ---------------------------
# Source docs
# ---------------------------

def source_doc_exists(conn, file_hash: str, entity_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM source_documents
            WHERE file_hash = %s AND entity_id = %s
            LIMIT 1
            """,
            (file_hash, entity_id),
        )
        return cur.fetchone() is not None


def insert_source_document(
    conn,
    *,
    doc_id: str,
    entity_id: str,
    entity_type: str,
    title: str,
    file_path: str,
    file_mtime,
    file_hash: str,
    doc_date,
    extracted_text: str,
):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO source_documents
            (doc_id, entity_id, entity_type, doc_type, title, file_path, file_mtime, file_hash, doc_date, extracted_text)
            VALUES (%s,%s,%s,'local_pdf',%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                doc_id,
                entity_id,
                entity_type,
                title,
                file_path,
                file_mtime,
                file_hash,
                doc_date,
                extracted_text,
            ),
        )
    conn.commit()


def insert_source_document_chunks(
    conn,
    *,
    doc_id: str,
    entity_id: str,
    chunks: list[str],
):
    with conn.cursor() as cur:
        for idx, chunk_text in enumerate(chunks):
            cur.execute(
                """
                INSERT INTO source_document_chunks (doc_id, entity_id, chunk_index, chunk_text)
                VALUES (%s,%s,%s,%s)
                """,
                (doc_id, entity_id, idx, chunk_text),
            )
    conn.commit()


def company_has_docs(conn, entity_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM source_documents WHERE entity_id=%s LIMIT 1
            """,
            (entity_id,),
        )
        return cur.fetchone() is not None


def get_recent_doc_ids(conn, entity_id: str, limit: int = 3) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id
            FROM source_documents
            WHERE entity_id = %s
            ORDER BY doc_date DESC NULLS LAST, created_at DESC
            LIMIT %s
            """,
            (entity_id, limit),
        )
        return [r[0] for r in cur.fetchall()]


def get_chunks_for_docs(conn, entity_id: str, doc_ids: list[str]) -> list[str]:
    if not doc_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_text
            FROM source_document_chunks
            WHERE entity_id=%s AND doc_id = ANY(%s)
            """,
            (entity_id, doc_ids),
        )
        return [r[0] for r in cur.fetchall()]
