import sqlite3

from app.services.metadata import MetadataStore


def test_initialize_migrates_legacy_documents_table(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                source_key TEXT NOT NULL UNIQUE,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_url TEXT,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )

    store = MetadataStore(database)
    store.initialize()

    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
    assert "embedding_fingerprint" in columns
