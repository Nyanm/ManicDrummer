"""Open / read / write access to manic.sqlite. Readers open read-only (can never mutate the corpus);
the writer (re)creates the schema on a fresh file and inserts rows in one transaction."""
import sqlite3
from pathlib import Path

from .blob import unpack_ir
from .schema import COLUMNS, SCHEMA

_INSERT = f"INSERT OR REPLACE INTO charts VALUES ({','.join('?' * len(COLUMNS))})"
_META_COLUMNS = [column for column in COLUMNS if column not in ("ir_rich", "ir_grid")]


def connect_ro(path_db) -> sqlite3.Connection:
    """Open manic.sqlite read-only (URI mode=ro) so consumers can never write to it"""
    return sqlite3.connect(f"{Path(path_db).resolve().as_uri()}?mode=ro", uri=True)


def connect_rw(path_db) -> sqlite3.Connection:
    """Open (or create) manic.sqlite read-write, for the writer"""
    return sqlite3.connect(str(path_db))


def write_charts(path_db, rows) -> int:
    """(Re)create the schema on a FRESH db file and insert every row (tuple in COLUMNS order) in one
    transaction; returns the row count. The db is a regenerable derived artifact, so an existing file
    is replaced."""
    path_db = Path(path_db)
    if path_db.exists():
        path_db.unlink()
    conn = connect_rw(path_db)
    try:
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute(SCHEMA)
        count = 0
        with conn:  # one transaction
            for row in rows:
                conn.execute(_INSERT, row)
                count += 1
        return count
    finally:
        conn.close()


def _read_blob(conn: sqlite3.Connection, chart_id: str, column: str):
    row = conn.execute(f"SELECT {column} FROM charts WHERE chart_id = ?", (chart_id,)).fetchone()
    if row is None:
        raise KeyError(chart_id)
    return unpack_ir(row[0])


def read_ir_rich(conn: sqlite3.Connection, chart_id: str) -> dict:
    """Decode one chart's ir_rich blob (timing + keysounds + notes) by chart_id"""
    return _read_blob(conn, chart_id, "ir_rich")


def read_ir_grid(conn: sqlite3.Connection, chart_id: str) -> dict:
    """Decode one chart's ir_grid blob (grid-quantised note rows) by chart_id"""
    return _read_blob(conn, chart_id, "ir_grid")


def read_meta(conn: sqlite3.Connection, chart_id: str) -> dict:
    """One chart's metadata columns (everything but the two IR blobs) as a {column: value} dict.
    Schema knowledge stays here, so callers read named fields instead of unpacking a positional row."""
    row = conn.execute(f"SELECT {','.join(_META_COLUMNS)} FROM charts WHERE chart_id = ?", (chart_id,)).fetchone()
    if row is None:
        raise KeyError(chart_id)
    return dict(zip(_META_COLUMNS, row))
