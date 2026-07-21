from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class StoredDocument:
    id: str
    source_key: str
    source_name: str
    source_type: str
    source_url: str | None
    content_hash: str
    embedding_fingerprint: str | None
    chunk_count: int
    updated_at: datetime


@dataclass(frozen=True)
class IngestionRun:
    id: str
    kind: str
    source_label: str
    status: str
    documents_processed: int
    chunks_created: int
    error_message: str | None
    created_at: datetime
    finished_at: datetime | None


class MetadataStore:
    """SQLite catalogue that defines which Chroma vectors are currently active."""

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_key TEXT NOT NULL UNIQUE,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_url TEXT,
                    content_hash TEXT NOT NULL,
                    embedding_fingerprint TEXT,
                    chunk_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_chunks (
                    vector_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    content_hash TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    UNIQUE(document_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    documents_processed INTEGER NOT NULL DEFAULT 0,
                    chunks_created INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT
                );
                """
            )
            columns = {
                str(row["name"]) for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "embedding_fingerprint" not in columns:
                connection.execute("ALTER TABLE documents ADD COLUMN embedding_fingerprint TEXT")

    def document_for_source(self, source_key: str) -> StoredDocument | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT id, source_key, source_name, source_type, source_url,
                          content_hash, embedding_fingerprint, chunk_count, updated_at
                   FROM documents WHERE source_key = ?""",
                (source_key,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def active_vector_ids_for_source(self, source_key: str) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT c.vector_id FROM document_chunks c
                   JOIN documents d ON d.id = c.document_id
                   WHERE d.source_key = ?""",
                (source_key,),
            ).fetchall()
        return [str(row["vector_id"]) for row in rows]

    def replace_document(self, document: StoredDocument, vector_ids: list[str]) -> None:
        if len(vector_ids) != document.chunk_count:
            raise ValueError("Each indexed chunk requires one vector ID.")
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO documents (
                       id, source_key, source_name, source_type, source_url,
                       content_hash, embedding_fingerprint, chunk_count, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_key) DO UPDATE SET
                       source_name = excluded.source_name,
                       source_type = excluded.source_type,
                       source_url = excluded.source_url,
                       content_hash = excluded.content_hash,
                       embedding_fingerprint = excluded.embedding_fingerprint,
                       chunk_count = excluded.chunk_count,
                       updated_at = excluded.updated_at""",
                (
                    document.id,
                    document.source_key,
                    document.source_name,
                    document.source_type,
                    document.source_url,
                    document.content_hash,
                    document.embedding_fingerprint,
                    document.chunk_count,
                    document.updated_at.isoformat(),
                ),
            )
            connection.execute("DELETE FROM document_chunks WHERE document_id = ?", (document.id,))
            connection.executemany(
                """INSERT INTO document_chunks
                   (vector_id, document_id, content_hash, chunk_index)
                   VALUES (?, ?, ?, ?)""",
                [
                    (vector_id, document.id, document.content_hash, chunk_index)
                    for chunk_index, vector_id in enumerate(vector_ids)
                ],
            )

    def active_vector_ids(self, vector_ids: list[str]) -> set[str]:
        if not vector_ids:
            return set()
        placeholders = ", ".join("?" for _ in vector_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT vector_id FROM document_chunks WHERE vector_id IN ({placeholders})",
                vector_ids,
            ).fetchall()
        return {str(row["vector_id"]) for row in rows}

    def list_documents(self, limit: int = 100) -> list[StoredDocument]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, source_key, source_name, source_type, source_url,
                          content_hash, embedding_fingerprint, chunk_count, updated_at
                   FROM documents ORDER BY updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def counts(self) -> tuple[int, int]:
        with self._connection() as connection:
            document_count = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0])
        return document_count, chunk_count

    def create_run(self, kind: str, source_label: str) -> IngestionRun:
        run = IngestionRun(
            id=str(uuid.uuid4()),
            kind=kind,
            source_label=source_label[:500],
            status="running",
            documents_processed=0,
            chunks_created=0,
            error_message=None,
            created_at=datetime.now(timezone.utc),
            finished_at=None,
        )
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO ingestion_runs
                   (id, kind, source_label, status, documents_processed, chunks_created,
                    error_message, created_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.id,
                    run.kind,
                    run.source_label,
                    run.status,
                    0,
                    0,
                    None,
                    run.created_at.isoformat(),
                    None,
                ),
            )
        return run

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        documents_processed: int = 0,
        chunks_created: int = 0,
        error_message: str | None = None,
    ) -> None:
        finished_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """UPDATE ingestion_runs SET status = ?, documents_processed = ?,
                   chunks_created = ?, error_message = ?, finished_at = ? WHERE id = ?""",
                (status, documents_processed, chunks_created, error_message, finished_at, run_id),
            )

    def recent_runs(self, limit: int = 10) -> list[IngestionRun]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, kind, source_label, status, documents_processed,
                          chunks_created, error_message, created_at, finished_at
                   FROM ingestion_runs ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    @staticmethod
    def stable_document_id(source_key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, source_key))

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if value is None:
            return None
        result = datetime.fromisoformat(value)
        return result if result.tzinfo else result.replace(tzinfo=timezone.utc)

    @classmethod
    def _row_to_document(cls, row: sqlite3.Row) -> StoredDocument:
        return StoredDocument(
            id=str(row["id"]),
            source_key=str(row["source_key"]),
            source_name=str(row["source_name"]),
            source_type=str(row["source_type"]),
            source_url=str(row["source_url"]) if row["source_url"] else None,
            content_hash=str(row["content_hash"]),
            embedding_fingerprint=(
                str(row["embedding_fingerprint"]) if row["embedding_fingerprint"] else None
            ),
            chunk_count=int(row["chunk_count"]),
            updated_at=cls._parse_datetime(str(row["updated_at"])) or datetime.now(timezone.utc),
        )

    @classmethod
    def _row_to_run(cls, row: sqlite3.Row) -> IngestionRun:
        return IngestionRun(
            id=str(row["id"]),
            kind=str(row["kind"]),
            source_label=str(row["source_label"]),
            status=str(row["status"]),
            documents_processed=int(row["documents_processed"]),
            chunks_created=int(row["chunks_created"]),
            error_message=str(row["error_message"]) if row["error_message"] else None,
            created_at=cls._parse_datetime(str(row["created_at"])) or datetime.now(timezone.utc),
            finished_at=cls._parse_datetime(str(row["finished_at"])) if row["finished_at"] else None,
        )
