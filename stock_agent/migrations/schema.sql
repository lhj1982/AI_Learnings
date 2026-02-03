-- Daily analysis table
CREATE TABLE IF NOT EXISTS stock_daily_analysis (
  date DATE NOT NULL,
  entity_id TEXT NOT NULL,
  entity_type TEXT NOT NULL,          -- public | private
  company_overview TEXT,
  recent_financials TEXT,
  valuation_metrics TEXT,
  technical_analysis TEXT,
  catalysts TEXT,
  risks TEXT,
  recommendation TEXT,
  raw_context JSONB,
  created_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY (date, entity_id)
);

-- Source documents table (local PDFs)
CREATE TABLE IF NOT EXISTS source_documents (
  doc_id TEXT PRIMARY KEY,            -- sha256(file)
  entity_id TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  doc_type TEXT NOT NULL,             -- local_pdf
  title TEXT,
  file_path TEXT NOT NULL,
  file_mtime TIMESTAMP,
  file_hash TEXT,
  doc_date DATE,                      -- inferred from filename, else file_mtime date
  extracted_text TEXT,
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_source_docs_entity
  ON source_documents(entity_id);

CREATE INDEX IF NOT EXISTS idx_source_docs_entity_docdate
  ON source_documents(entity_id, doc_date DESC);

-- Chunk table for retrieval
CREATE TABLE IF NOT EXISTS source_document_chunks (
  chunk_id BIGSERIAL PRIMARY KEY,
  doc_id TEXT NOT NULL REFERENCES source_documents(doc_id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL,
  chunk_index INT NOT NULL,
  chunk_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_entity
  ON source_document_chunks(entity_id);
