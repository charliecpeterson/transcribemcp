import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 8

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meetings (
    id               TEXT PRIMARY KEY,
    project_id       TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title            TEXT NOT NULL,
    date             TEXT,
    duration_seconds INTEGER,
    audio_path       TEXT NOT NULL,
    transcript_path  TEXT,
    status           TEXT NOT NULL DEFAULT 'pending',
    backend          TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_meetings_project ON meetings(project_id);

CREATE TABLE IF NOT EXISTS speakers (
    id         TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    name       TEXT,
    person_id  TEXT,
    notes      TEXT,
    UNIQUE(meeting_id, label)
);
CREATE INDEX IF NOT EXISTS idx_speakers_meeting ON speakers(meeting_id);

CREATE TABLE IF NOT EXISTS chunks (
    id         TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker_id TEXT REFERENCES speakers(id) ON DELETE SET NULL,
    text       TEXT NOT NULL,
    start_time REAL,
    end_time   REAL
);
CREATE INDEX IF NOT EXISTS idx_chunks_meeting ON chunks(meeting_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    meeting_id   TEXT REFERENCES meetings(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL,
    stage        TEXT,
    progress     REAL,
    error        TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_meeting ON jobs(meeting_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    meeting_id  TEXT REFERENCES meetings(id) ON DELETE SET NULL,
    title       TEXT NOT NULL,
    kind        TEXT NOT NULL,
    source_path TEXT,
    char_count  INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
CREATE INDEX IF NOT EXISTS idx_documents_meeting ON documents(meeting_id);

CREATE TABLE IF NOT EXISTS document_chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    UNIQUE(document_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document ON document_chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    text,
    content='document_chunks',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS doc_chunks_ai AFTER INSERT ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS doc_chunks_ad AFTER DELETE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS doc_chunks_au AFTER UPDATE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO document_chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""

SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS persons (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    email      TEXT,
    role       TEXT,
    notes      TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);

-- speakers.person_id existed since v1 as a bare TEXT column; now that persons
-- is a real table, wire up ON DELETE SET NULL via a FK-aware index and a
-- manual trigger (SQLite can't alter a column to add a FK in-place).
CREATE TRIGGER IF NOT EXISTS persons_ad AFTER DELETE ON persons BEGIN
    UPDATE speakers SET person_id = NULL WHERE person_id = old.id;
END;
"""

SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS meeting_series (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_meeting_series_project ON meeting_series(project_id);

CREATE TABLE IF NOT EXISTS meeting_series_members (
    series_id  TEXT NOT NULL REFERENCES meeting_series(id) ON DELETE CASCADE,
    meeting_id TEXT NOT NULL REFERENCES meetings(id)       ON DELETE CASCADE,
    added_at   TEXT NOT NULL,
    PRIMARY KEY (series_id, meeting_id)
);
CREATE INDEX IF NOT EXISTS idx_msm_meeting ON meeting_series_members(meeting_id);
"""

SCHEMA_V5 = """
CREATE TABLE IF NOT EXISTS summaries (
    id         TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(meeting_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_summaries_meeting ON summaries(meeting_id);
CREATE INDEX IF NOT EXISTS idx_summaries_kind    ON summaries(kind);

CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
    text,
    content='summaries',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries BEGIN
    INSERT INTO summaries_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON summaries BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE ON summaries BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO summaries_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""

SCHEMA_V6 = """
-- Summaries become scopable to either a single meeting OR a series, not both.
-- meeting_id is relaxed to nullable; series_id is added; a CHECK enforces XOR.
-- We rebuild the table (SQLite can't relax NOT NULL or add a FK in place),
-- copy data forward with series_id = NULL, then rebuild the FTS index.

DROP TRIGGER IF EXISTS summaries_ai;
DROP TRIGGER IF EXISTS summaries_ad;
DROP TRIGGER IF EXISTS summaries_au;
DROP TABLE IF EXISTS summaries_fts;

CREATE TABLE summaries_new (
    id         TEXT PRIMARY KEY,
    meeting_id TEXT REFERENCES meetings(id)       ON DELETE CASCADE,
    series_id  TEXT REFERENCES meeting_series(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        (meeting_id IS NOT NULL AND series_id IS NULL)
     OR (meeting_id IS NULL AND series_id IS NOT NULL)
    )
);

INSERT INTO summaries_new(id, meeting_id, series_id, kind, text, created_at, updated_at)
    SELECT id, meeting_id, NULL, kind, text, created_at, updated_at FROM summaries;

DROP TABLE summaries;
ALTER TABLE summaries_new RENAME TO summaries;

CREATE INDEX idx_summaries_meeting ON summaries(meeting_id);
CREATE INDEX idx_summaries_series  ON summaries(series_id);
CREATE INDEX idx_summaries_kind    ON summaries(kind);
CREATE UNIQUE INDEX idx_summaries_meeting_kind
    ON summaries(meeting_id, kind) WHERE meeting_id IS NOT NULL;
CREATE UNIQUE INDEX idx_summaries_series_kind
    ON summaries(series_id, kind)  WHERE series_id  IS NOT NULL;

CREATE VIRTUAL TABLE summaries_fts USING fts5(
    text,
    content='summaries',
    content_rowid='rowid'
);
INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild');

CREATE TRIGGER summaries_ai AFTER INSERT ON summaries BEGIN
    INSERT INTO summaries_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER summaries_ad AFTER DELETE ON summaries BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER summaries_au AFTER UPDATE ON summaries BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO summaries_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""

SCHEMA_V7 = """
-- Two small columns that answer long-standing UX problems:
--   jobs.progress_updated_at  — heartbeat so callers can tell "stuck" from
--                               "slow" even when the progress number hasn't
--                               moved (whisperx sits at asr 0.15 for ages).
--   summaries.transcript_stale — flag flipped by retranscribe_meeting so
--                               callers know a saved summary may describe
--                               an older version of the transcript.
ALTER TABLE jobs ADD COLUMN progress_updated_at TEXT;
ALTER TABLE summaries ADD COLUMN transcript_stale INTEGER NOT NULL DEFAULT 0;
"""

SCHEMA_V8 = """
-- Checkpoint in seconds of audio timeline: the last window that was fully
-- transcribed + persisted by this job. resume_job picks up by re-planning
-- VAD windows (deterministic) and skipping any with end <= checkpoint.
--   0.0 means "no windows done" (fresh start).
ALTER TABLE jobs ADD COLUMN checkpoint_seconds REAL NOT NULL DEFAULT 0;
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_one_of(
    meeting_id: str | None,
    series_id: str | None,
    *,
    label: str = "scope",
) -> None:
    """Validate that exactly one of (meeting_id, series_id) is provided.

    Used by anything that writes or reads a per-scope artifact (summaries,
    chat context bundles) where the two scopes are mutually exclusive at the
    schema level.
    """
    if (meeting_id is None) == (series_id is None):
        raise ValueError(
            f"exactly one of meeting_id or series_id must be provided ({label})"
        )


def new_id() -> str:
    return uuid.uuid4().hex


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < 1:
        conn.executescript(SCHEMA_V1)
    if current < 2:
        conn.executescript(SCHEMA_V2)
    if current < 3:
        conn.executescript(SCHEMA_V3)
    if current < 4:
        conn.executescript(SCHEMA_V4)
    if current < 5:
        conn.executescript(SCHEMA_V5)
    if current < 6:
        conn.executescript(SCHEMA_V6)
    if current < 7:
        conn.executescript(SCHEMA_V7)
    if current < 8:
        conn.executescript(SCHEMA_V8)
    if current < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    # Reconcile any jobs left in-flight from a crashed previous run.
    conn.execute(
        "UPDATE jobs SET status = 'error', error = COALESCE(error, 'interrupted'), "
        "finished_at = ? WHERE status IN ('queued', 'running')",
        (now_iso(),),
    )
    # Any meeting stuck mid-transcribe goes back to error.
    conn.execute(
        "UPDATE meetings SET status = 'error' WHERE status = 'transcribing'"
    )
    return conn


# Per-thread SQLite connection. Python's sqlite3 with check_same_thread=False
# disables the thread guard but does NOT serialize concurrent execute() calls
# on one connection — iterator state can interleave under FastMCP if two tool
# handlers run on different threads. One connection per thread sidesteps that;
# WAL mode handles inter-connection contention at the file level.
_local = threading.local()

# Tests reuse a single connection and inject it via reset_conn_for_tests. When
# set, all callers on every thread see this connection (which is what the test
# fixture wants — single source of truth for the assertion thread). Production
# leaves this None and uses the thread-local path.
_test_override: sqlite3.Connection | None = None

# Migrations + crash reconciliation must run exactly once per process. If
# every thread's first get_conn() re-ran init(), a thread joining while a
# real job was running would wrongly flip that job's status to 'error'.
_init_lock = threading.Lock()
_db_path: Path | None = None
_migrations_done = False


def _ensure_migrations_run() -> Path:
    global _db_path, _migrations_done
    with _init_lock:
        if _migrations_done:
            return _db_path  # type: ignore[return-value]
        from .config import get_settings
        _db_path = get_settings().db_path
        # init() opens its own connection just to run the migration ladder
        # and reconciliation; we close it and the per-thread connections
        # below use plain connect() to skip that work.
        bootstrap = init(_db_path)
        bootstrap.close()
        _migrations_done = True
        return _db_path


def get_conn() -> sqlite3.Connection:
    if _test_override is not None:
        return _test_override
    db_path = _ensure_migrations_run()
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = connect(db_path)
        _local.conn = conn
    return conn


def reset_conn_for_tests(conn: sqlite3.Connection | None) -> None:
    """Test hook: route every get_conn() call to `conn` (across all threads).
    Pass None to disable the override."""
    global _test_override
    _test_override = conn


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit transaction (we run with isolation_level=None / autocommit)."""
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
