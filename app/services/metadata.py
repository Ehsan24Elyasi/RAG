from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.db.migrations import DEFAULT_WORKSPACE_ID, run_migrations


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


@dataclass(frozen=True)
class Workspace:
    id: str
    slug: str
    name: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class User:
    id: str
    workspace_id: str
    external_id: str
    display_name: str | None
    email: str | None
    metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Conversation:
    id: str
    workspace_id: str
    user_id: str
    title: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None


@dataclass(frozen=True)
class Message:
    id: str
    workspace_id: str
    conversation_id: str
    user_id: str | None
    role: str
    status: str
    content: str | None
    client_message_id: str | None
    parent_message_id: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True)
class MessageCitation:
    id: str
    message_id: str
    document_id: str | None
    vector_id: str | None
    source_name: str | None
    source_url: str | None
    excerpt: str | None
    position: int
    metadata: dict[str, Any] | None
    created_at: datetime


@dataclass(frozen=True)
class GenerationRun:
    id: str
    workspace_id: str
    conversation_id: str
    message_id: str
    provider: str | None
    model: str | None
    status: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: int | None
    finish_reason: str | None
    request_id: str | None
    metadata: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    finished_at: datetime | None


class MetadataStore:
    """Explicit SQLite repositories for indexed content and persisted conversations."""

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
        with sqlite3.connect(self.database_path, timeout=30, isolation_level=None) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            run_migrations(connection)

    def default_workspace(self) -> Workspace:
        workspace = self.workspace(DEFAULT_WORKSPACE_ID)
        if workspace is None:
            raise RuntimeError("MetadataStore.initialize() must be called first.")
        return workspace

    def workspace(self, workspace_id: str) -> Workspace | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT id, slug, name, created_at, updated_at
                   FROM workspaces WHERE id = ?""",
                (workspace_id,),
            ).fetchone()
        return self._row_to_workspace(row) if row else None

    def document_for_source(self, source_key: str) -> StoredDocument | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT id, source_key, source_name, source_type, source_url,
                          content_hash, embedding_fingerprint, chunk_count, updated_at
                   FROM documents WHERE workspace_id = ? AND source_key = ?""",
                (DEFAULT_WORKSPACE_ID, source_key),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def active_vector_ids_for_source(self, source_key: str) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT c.vector_id FROM document_chunks c
                   JOIN documents d ON d.id = c.document_id
                   WHERE d.workspace_id = ? AND d.source_key = ?""",
                (DEFAULT_WORKSPACE_ID, source_key),
            ).fetchall()
        return [str(row["vector_id"]) for row in rows]

    def replace_document(self, document: StoredDocument, vector_ids: list[str]) -> None:
        if len(vector_ids) != document.chunk_count:
            raise ValueError("Each indexed chunk requires one vector ID.")
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO documents (
                       id, workspace_id, source_key, source_name, source_type, source_url,
                       content_hash, embedding_fingerprint, chunk_count, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(workspace_id, source_key) DO UPDATE SET
                       source_name = excluded.source_name,
                       source_type = excluded.source_type,
                       source_url = excluded.source_url,
                       content_hash = excluded.content_hash,
                       embedding_fingerprint = excluded.embedding_fingerprint,
                       chunk_count = excluded.chunk_count,
                       updated_at = excluded.updated_at""",
                (
                    document.id,
                    DEFAULT_WORKSPACE_ID,
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
                   FROM documents WHERE workspace_id = ?
                   ORDER BY updated_at DESC LIMIT ?""",
                (DEFAULT_WORKSPACE_ID, limit),
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def counts(self) -> tuple[int, int]:
        with self._connection() as connection:
            document_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE workspace_id = ?",
                    (DEFAULT_WORKSPACE_ID,),
                ).fetchone()[0]
            )
            chunk_count = int(
                connection.execute(
                    """SELECT COUNT(*) FROM document_chunks c
                       JOIN documents d ON d.id = c.document_id WHERE d.workspace_id = ?""",
                    (DEFAULT_WORKSPACE_ID,),
                ).fetchone()[0]
            )
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
            created_at=self._now(),
            finished_at=None,
        )
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO ingestion_runs
                   (id, workspace_id, kind, source_label, status, documents_processed,
                    chunks_created, error_message, created_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.id,
                    DEFAULT_WORKSPACE_ID,
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
        finished_at = self._now().isoformat()
        with self._connection() as connection:
            connection.execute(
                """UPDATE ingestion_runs SET status = ?, documents_processed = ?,
                   chunks_created = ?, error_message = ?, finished_at = ?
                   WHERE workspace_id = ? AND id = ?""",
                (
                    status,
                    documents_processed,
                    chunks_created,
                    error_message,
                    finished_at,
                    DEFAULT_WORKSPACE_ID,
                    run_id,
                ),
            )

    def recent_runs(self, limit: int = 10) -> list[IngestionRun]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, kind, source_label, status, documents_processed,
                          chunks_created, error_message, created_at, finished_at
                   FROM ingestion_runs WHERE workspace_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (DEFAULT_WORKSPACE_ID, limit),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def upsert_user(
        self,
        workspace_id: str,
        external_id: str,
        *,
        display_name: str | None = None,
        email: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> User:
        now = self._now().isoformat()
        user_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{workspace_id}:{external_id}"))
        metadata_json = self._dump_json(metadata)
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO users
                   (id, workspace_id, external_id, display_name, email, metadata_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(workspace_id, external_id) DO UPDATE SET
                       display_name = excluded.display_name,
                       email = excluded.email,
                       metadata_json = excluded.metadata_json,
                       updated_at = excluded.updated_at""",
                (
                    user_id,
                    workspace_id,
                    external_id,
                    display_name,
                    email,
                    metadata_json,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """SELECT * FROM users WHERE workspace_id = ? AND external_id = ?""",
                (workspace_id, external_id),
            ).fetchone()
        assert row is not None
        return self._row_to_user(row)

    def create_conversation(
        self, workspace_id: str, user_id: str, *, title: str | None = None
    ) -> Conversation:
        conversation_id = str(uuid.uuid4())
        now = self._now().isoformat()
        with self._connection() as connection:
            authorized = connection.execute(
                "SELECT 1 FROM users WHERE id = ? AND workspace_id = ?",
                (user_id, workspace_id),
            ).fetchone()
            if authorized is None:
                raise ValueError("User does not belong to the workspace.")
            connection.execute(
                """INSERT INTO conversations
                   (id, workspace_id, user_id, title, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'active', ?, ?)""",
                (conversation_id, workspace_id, user_id, title, now, now),
            )
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND workspace_id = ?",
                (conversation_id, workspace_id),
            ).fetchone()
        assert row is not None
        return self._row_to_conversation(row)

    def get_conversation(
        self, workspace_id: str, conversation_id: str, *, user_id: str | None = None
    ) -> Conversation | None:
        sql = "SELECT * FROM conversations WHERE workspace_id = ? AND id = ?"
        parameters: list[str] = [workspace_id, conversation_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            parameters.append(user_id)
        with self._connection() as connection:
            row = connection.execute(sql, parameters).fetchone()
        return self._row_to_conversation(row) if row else None

    def list_authorized_conversations(
        self, workspace_id: str, user_id: str, *, limit: int = 50
    ) -> list[Conversation]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM conversations
                   WHERE workspace_id = ? AND user_id = ?
                   ORDER BY updated_at DESC, id DESC LIMIT ?""",
                (workspace_id, user_id, limit),
            ).fetchall()
        return [self._row_to_conversation(row) for row in rows]

    def append_user_message(
        self,
        workspace_id: str,
        conversation_id: str,
        user_id: str,
        content: str,
        *,
        client_message_id: str,
    ) -> Message:
        now = self._now().isoformat()
        with self._connection() as connection:
            conversation = self._authorized_conversation(
                connection, workspace_id, conversation_id, user_id
            )
            if conversation is None:
                raise ValueError("Conversation is not authorized for this user and workspace.")
            message_id = str(uuid.uuid4())
            cursor = connection.execute(
                """INSERT OR IGNORE INTO messages
                   (id, workspace_id, conversation_id, user_id, role, status, content,
                    client_message_id, created_at, updated_at, completed_at)
                   VALUES (?, ?, ?, ?, 'user', 'completed', ?, ?, ?, ?, ?)""",
                (
                    message_id,
                    workspace_id,
                    conversation_id,
                    user_id,
                    content,
                    client_message_id,
                    now,
                    now,
                    now,
                ),
            )
            if cursor.rowcount:
                self._touch_conversation(connection, workspace_id, conversation_id, now)
            row = connection.execute(
                """SELECT * FROM messages WHERE workspace_id = ? AND conversation_id = ?
                   AND client_message_id = ?""",
                (workspace_id, conversation_id, client_message_id),
            ).fetchone()
        assert row is not None
        return self._row_to_message(row)

    def create_assistant_placeholder(
        self,
        workspace_id: str,
        conversation_id: str,
        *,
        user_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> Message:
        message, _ = self.claim_assistant_placeholder(
            workspace_id,
            conversation_id,
            user_id=user_id,
            parent_message_id=parent_message_id,
        )
        return message

    def claim_assistant_placeholder(
        self,
        workspace_id: str,
        conversation_id: str,
        *,
        user_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> tuple[Message, bool]:
        now = self._now().isoformat()
        message_id = str(uuid.uuid4())
        with self._connection() as connection:
            conversation = connection.execute(
                """SELECT 1 FROM conversations WHERE workspace_id = ? AND id = ?
                   AND (? IS NULL OR user_id = ?)""",
                (workspace_id, conversation_id, user_id, user_id),
            ).fetchone()
            if conversation is None:
                raise ValueError("Conversation is not authorized in this workspace.")
            if parent_message_id:
                parent = connection.execute(
                    """SELECT 1 FROM messages WHERE workspace_id = ? AND conversation_id = ?
                       AND id = ? AND role = 'user'""",
                    (workspace_id, conversation_id, parent_message_id),
                ).fetchone()
                if parent is None:
                    raise ValueError("Parent user message does not exist in this conversation.")
                existing = connection.execute(
                    """SELECT * FROM messages WHERE workspace_id = ?
                       AND parent_message_id = ? AND role = 'assistant'""",
                    (workspace_id, parent_message_id),
                ).fetchone()
                if existing is not None:
                    return self._row_to_message(existing), False
            cursor = connection.execute(
                """INSERT OR IGNORE INTO messages
                   (id, workspace_id, conversation_id, role, status, parent_message_id,
                    created_at, updated_at)
                   VALUES (?, ?, ?, 'assistant', 'generating', ?, ?, ?)""",
                (message_id, workspace_id, conversation_id, parent_message_id, now, now),
            )
            if cursor.rowcount:
                self._touch_conversation(connection, workspace_id, conversation_id, now)
            if parent_message_id:
                row = connection.execute(
                    """SELECT * FROM messages WHERE workspace_id = ?
                       AND parent_message_id = ? AND role = 'assistant'""",
                    (workspace_id, parent_message_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM messages WHERE workspace_id = ? AND id = ?",
                    (workspace_id, message_id),
                ).fetchone()
        assert row is not None
        return self._row_to_message(row), bool(cursor.rowcount)

    def complete_assistant_message(
        self, workspace_id: str, message_id: str, content: str
    ) -> Message | None:
        return self._finish_assistant_message(workspace_id, message_id, "completed", content, None)

    def fail_assistant_message(
        self, workspace_id: str, message_id: str, error_message: str
    ) -> Message | None:
        return self._finish_assistant_message(
            workspace_id, message_id, "failed", None, error_message
        )

    def stop_assistant_message(
        self, workspace_id: str, message_id: str, content: str | None = None
    ) -> Message | None:
        return self._finish_assistant_message(workspace_id, message_id, "stopped", content, None)

    def recover_stale_generating_messages(self, stale_after_seconds: int) -> int:
        cutoff = (self._now() - timedelta(seconds=stale_after_seconds)).isoformat()
        now = self._now().isoformat()
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE messages SET status = 'failed',
                       error_message = 'Generation was interrupted.', updated_at = ?, completed_at = ?
                   WHERE role = 'assistant' AND status = 'generating' AND updated_at < ?""",
                (now, now, cutoff),
            )
        return cursor.rowcount

    def load_conversation_history(
        self, workspace_id: str, conversation_id: str, *, user_id: str, limit: int = 50
    ) -> list[Message]:
        if limit <= 0:
            return []
        with self._connection() as connection:
            if self._authorized_conversation(
                connection, workspace_id, conversation_id, user_id
            ) is None:
                return []
            rows = connection.execute(
                """SELECT * FROM (
                       SELECT * FROM messages
                       WHERE workspace_id = ? AND conversation_id = ?
                       ORDER BY created_at DESC, id DESC LIMIT ?
                   ) ORDER BY created_at ASC, id ASC""",
                (workspace_id, conversation_id, limit),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def record_citations(
        self,
        workspace_id: str,
        message_id: str,
        citations: Sequence[Mapping[str, Any]],
    ) -> list[MessageCitation]:
        now = self._now().isoformat()
        with self._connection() as connection:
            message = connection.execute(
                """SELECT 1 FROM messages WHERE workspace_id = ? AND id = ?
                   AND role = 'assistant'""",
                (workspace_id, message_id),
            ).fetchone()
            if message is None:
                raise ValueError("Assistant message does not exist in this workspace.")
            connection.execute(
                "DELETE FROM message_citations WHERE workspace_id = ? AND message_id = ?",
                (workspace_id, message_id),
            )
            for position, citation in enumerate(citations):
                connection.execute(
                    """INSERT INTO message_citations
                       (id, workspace_id, message_id, document_id, vector_id, source_name,
                        source_url, excerpt, position, metadata_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        workspace_id,
                        message_id,
                        citation.get("document_id"),
                        citation.get("vector_id"),
                        citation.get("source_name"),
                        citation.get("source_url"),
                        citation.get("excerpt"),
                        position,
                        self._dump_json(citation.get("metadata")),
                        now,
                    ),
                )
            rows = connection.execute(
                """SELECT * FROM message_citations
                   WHERE workspace_id = ? AND message_id = ? ORDER BY position""",
                (workspace_id, message_id),
            ).fetchall()
        return [self._row_to_citation(row) for row in rows]

    def citations_for_message(
        self, workspace_id: str, message_id: str
    ) -> list[MessageCitation]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM message_citations
                   WHERE workspace_id = ? AND message_id = ? ORDER BY position""",
                (workspace_id, message_id),
            ).fetchall()
        return [self._row_to_citation(row) for row in rows]

    def record_generation_usage(
        self,
        workspace_id: str,
        conversation_id: str,
        message_id: str,
        *,
        status: str,
        provider: str | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: int | None = None,
        finish_reason: str | None = None,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> GenerationRun:
        run_id = str(uuid.uuid4())
        now = self._now().isoformat()
        finished_at = now if status != "running" else None
        with self._connection() as connection:
            message = connection.execute(
                """SELECT 1 FROM messages WHERE workspace_id = ? AND conversation_id = ?
                   AND id = ? AND role = 'assistant'""",
                (workspace_id, conversation_id, message_id),
            ).fetchone()
            if message is None:
                raise ValueError("Assistant message does not exist in this workspace conversation.")
            connection.execute(
                """INSERT INTO generation_runs
                   (id, workspace_id, conversation_id, message_id, provider, model, status,
                    prompt_tokens, completion_tokens, total_tokens, latency_ms, finish_reason,
                    request_id, metadata_json, error_message, created_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    workspace_id,
                    conversation_id,
                    message_id,
                    provider,
                    model,
                    status,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    latency_ms,
                    finish_reason,
                    request_id,
                    self._dump_json(metadata),
                    error_message,
                    now,
                    finished_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM generation_runs WHERE workspace_id = ? AND id = ?",
                (workspace_id, run_id),
            ).fetchone()
        assert row is not None
        return self._row_to_generation_run(row)

    def _finish_assistant_message(
        self,
        workspace_id: str,
        message_id: str,
        status: str,
        content: str | None,
        error_message: str | None,
    ) -> Message | None:
        now = self._now().isoformat()
        with self._connection() as connection:
            connection.execute(
                """UPDATE messages SET status = ?, content = COALESCE(?, content),
                   error_message = ?, updated_at = ?, completed_at = ?
                   WHERE workspace_id = ? AND id = ? AND role = 'assistant'""",
                (status, content, error_message, now, now, workspace_id, message_id),
            )
            row = connection.execute(
                """SELECT * FROM messages
                   WHERE workspace_id = ? AND id = ? AND role = 'assistant'""",
                (workspace_id, message_id),
            ).fetchone()
            if row is not None:
                self._touch_conversation(
                    connection, workspace_id, str(row["conversation_id"]), now
                )
        return self._row_to_message(row) if row else None

    @staticmethod
    def _authorized_conversation(
        connection: sqlite3.Connection,
        workspace_id: str,
        conversation_id: str,
        user_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """SELECT * FROM conversations
               WHERE workspace_id = ? AND id = ? AND user_id = ?""",
            (workspace_id, conversation_id, user_id),
        ).fetchone()

    @staticmethod
    def _touch_conversation(
        connection: sqlite3.Connection, workspace_id: str, conversation_id: str, now: str
    ) -> None:
        connection.execute(
            """UPDATE conversations SET updated_at = ?, last_message_at = ?
               WHERE workspace_id = ? AND id = ?""",
            (now, now, workspace_id, conversation_id),
        )

    @staticmethod
    def stable_document_id(source_key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, source_key))

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _dump_json(value: Mapping[str, Any] | Any | None) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _load_json(value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}

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
            updated_at=cls._parse_datetime(str(row["updated_at"])) or cls._now(),
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
            created_at=cls._parse_datetime(str(row["created_at"])) or cls._now(),
            finished_at=cls._parse_datetime(str(row["finished_at"])) if row["finished_at"] else None,
        )

    @classmethod
    def _row_to_workspace(cls, row: sqlite3.Row) -> Workspace:
        return Workspace(
            id=str(row["id"]),
            slug=str(row["slug"]),
            name=str(row["name"]),
            created_at=cls._parse_datetime(str(row["created_at"])) or cls._now(),
            updated_at=cls._parse_datetime(str(row["updated_at"])) or cls._now(),
        )

    @classmethod
    def _row_to_user(cls, row: sqlite3.Row) -> User:
        return User(
            id=str(row["id"]),
            workspace_id=str(row["workspace_id"]),
            external_id=str(row["external_id"]),
            display_name=str(row["display_name"]) if row["display_name"] else None,
            email=str(row["email"]) if row["email"] else None,
            metadata=cls._load_json(row["metadata_json"]),
            created_at=cls._parse_datetime(str(row["created_at"])) or cls._now(),
            updated_at=cls._parse_datetime(str(row["updated_at"])) or cls._now(),
        )

    @classmethod
    def _row_to_conversation(cls, row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=str(row["id"]),
            workspace_id=str(row["workspace_id"]),
            user_id=str(row["user_id"]),
            title=str(row["title"]) if row["title"] else None,
            status=str(row["status"]),
            created_at=cls._parse_datetime(str(row["created_at"])) or cls._now(),
            updated_at=cls._parse_datetime(str(row["updated_at"])) or cls._now(),
            last_message_at=(
                cls._parse_datetime(str(row["last_message_at"]))
                if row["last_message_at"]
                else None
            ),
        )

    @classmethod
    def _row_to_message(cls, row: sqlite3.Row) -> Message:
        return Message(
            id=str(row["id"]),
            workspace_id=str(row["workspace_id"]),
            conversation_id=str(row["conversation_id"]),
            user_id=str(row["user_id"]) if row["user_id"] else None,
            role=str(row["role"]),
            status=str(row["status"]),
            content=str(row["content"]) if row["content"] is not None else None,
            client_message_id=(
                str(row["client_message_id"]) if row["client_message_id"] else None
            ),
            parent_message_id=(
                str(row["parent_message_id"]) if row["parent_message_id"] else None
            ),
            error_message=str(row["error_message"]) if row["error_message"] else None,
            created_at=cls._parse_datetime(str(row["created_at"])) or cls._now(),
            updated_at=cls._parse_datetime(str(row["updated_at"])) or cls._now(),
            completed_at=(
                cls._parse_datetime(str(row["completed_at"])) if row["completed_at"] else None
            ),
        )

    @classmethod
    def _row_to_citation(cls, row: sqlite3.Row) -> MessageCitation:
        return MessageCitation(
            id=str(row["id"]),
            message_id=str(row["message_id"]),
            document_id=str(row["document_id"]) if row["document_id"] else None,
            vector_id=str(row["vector_id"]) if row["vector_id"] else None,
            source_name=str(row["source_name"]) if row["source_name"] else None,
            source_url=str(row["source_url"]) if row["source_url"] else None,
            excerpt=str(row["excerpt"]) if row["excerpt"] else None,
            position=int(row["position"]),
            metadata=cls._load_json(row["metadata_json"]),
            created_at=cls._parse_datetime(str(row["created_at"])) or cls._now(),
        )

    @classmethod
    def _row_to_generation_run(cls, row: sqlite3.Row) -> GenerationRun:
        return GenerationRun(
            id=str(row["id"]),
            workspace_id=str(row["workspace_id"]),
            conversation_id=str(row["conversation_id"]),
            message_id=str(row["message_id"]),
            provider=str(row["provider"]) if row["provider"] else None,
            model=str(row["model"]) if row["model"] else None,
            status=str(row["status"]),
            prompt_tokens=int(row["prompt_tokens"]) if row["prompt_tokens"] is not None else None,
            completion_tokens=(
                int(row["completion_tokens"])
                if row["completion_tokens"] is not None
                else None
            ),
            total_tokens=int(row["total_tokens"]) if row["total_tokens"] is not None else None,
            latency_ms=int(row["latency_ms"]) if row["latency_ms"] is not None else None,
            finish_reason=str(row["finish_reason"]) if row["finish_reason"] else None,
            request_id=str(row["request_id"]) if row["request_id"] else None,
            metadata=cls._load_json(row["metadata_json"]),
            error_message=str(row["error_message"]) if row["error_message"] else None,
            created_at=cls._parse_datetime(str(row["created_at"])) or cls._now(),
            finished_at=(
                cls._parse_datetime(str(row["finished_at"])) if row["finished_at"] else None
            ),
        )
