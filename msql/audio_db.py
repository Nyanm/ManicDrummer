"""The MuQ feature cache: manic_audio.sqlite (an INDEX) + a sibling manic_audio.features/ directory
holding one raw feature file per song. Ported from VFT's audited audio_db (file-per-audio layout:
a sqlite blob is an overflow page chain, so offset reads cost O(offset); a plain file preads flat).

Keys are music-id based: audio_key = "m{seq_id:04}" -- never a path (the audio_path five-way-sync
lesson). Feature files are `<audio_key>.f16`, frame-major [n_frame, n_layer, dim] STORE_DTYPE bytes
at the NATIVE 25Hz rate; beat alignment happens at read time (md.encode.beat_grid). The index row
carries the song's bpm anchors + end_beat so feature readers never open manic.sqlite.

Open ONE instance per process/scope -- a build opens one writer (open_write), each DataLoader
worker opens its own reader (open_read); sqlite connections must not cross threads/processes."""
import os
import sqlite3
from pathlib import Path

import msgpack
import numpy as np

FEATURE_DIR_SUFFIX = ".features"                          # manic_audio.sqlite -> manic_audio.features/
FEATURE_FILE_SUFFIX = ".f16"
FEATURE_LAYOUT = "file:frame_major[n_frame,n_layer,dim]"  # meta['layout']; a mismatch refuses to load
STORE_DTYPE = "float16"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_feature (
    audio_key   TEXT PRIMARY KEY,               -- "m{seq_id:04}"
    music_id    INTEGER NOT NULL,
    n_frame     INTEGER NOT NULL,
    n_layer     INTEGER NOT NULL,
    dim         INTEGER NOT NULL,
    feat_rel    TEXT NOT NULL,                  -- file name under <features>/
    timing_blob BLOB NOT NULL,                  -- msgpack [[beat, ms, bpm], ...] (beat in 1/480 units)
    end_beat    INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_audio_feat_rel ON audio_feature(feat_rel);
CREATE TABLE IF NOT EXISTS build_status (
    audio_key TEXT PRIMARY KEY,
    status    TEXT NOT NULL,                    -- 'done' | 'failed'
    reason    TEXT,
    seconds   REAL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def features_dir(path_db) -> Path:
    """The feature directory belonging to an index -- DERIVED, never configured separately, so it can
    never point at the wrong cache (the two must be moved together)."""
    return Path(path_db).with_suffix(FEATURE_DIR_SUFFIX)


class AudioDb:
    """Owns one sqlite connection to the index plus the feature directory path, and all reads/writes"""

    def __init__(self, conn: sqlite3.Connection, path_features: Path):
        self._conn = conn
        self._path_features = Path(path_features)

    @classmethod
    def open_write(cls, path_db) -> "AudioDb":
        conn = sqlite3.connect(str(path_db))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        db = cls(conn, features_dir(path_db))
        conn.executescript(_SCHEMA)
        conn.commit()
        db._path_features.mkdir(parents=True, exist_ok=True)
        return db

    @classmethod
    def open_read(cls, path_db) -> "AudioDb":
        """Read-only; refuses a cache whose meta['layout'] differs (verify-then-load, never repair)"""
        conn = sqlite3.connect(f"{Path(path_db).resolve().as_uri()}?mode=ro", uri=True)
        db = cls(conn, features_dir(path_db))
        str_layout = db.get_meta("layout")
        if str_layout != FEATURE_LAYOUT:
            raise ValueError(f"{path_db}: layout {str_layout!r} != {FEATURE_LAYOUT!r} -- rebuild the cache")
        return db

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- meta + resume manifest ----------------------------------------------------------------

    def set_meta(self, str_key: str, value) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (str_key, str(value)))

    def get_meta(self, str_key: str):
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (str_key,)).fetchone()
        return row[0] if row else None

    def is_done(self, str_audio_key: str) -> bool:
        return self._conn.execute("SELECT 1 FROM build_status WHERE audio_key = ? AND status = 'done'",
                                  (str_audio_key,)).fetchone() is not None

    def mark_done(self, str_audio_key: str, seconds: float) -> None:
        self._conn.execute("INSERT OR REPLACE INTO build_status VALUES (?, 'done', NULL, ?)",
                           (str_audio_key, seconds))

    def mark_failed(self, str_audio_key: str, str_reason: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO build_status VALUES (?, 'failed', ?, NULL)",
                           (str_audio_key, str_reason))

    # --- feature write / read ------------------------------------------------------------------

    def write_feature(self, str_audio_key: str, music_id: int, feat: np.ndarray,
                      vec_anchor: list, end_beat: int) -> None:
        """Store one song's [n_frame, n_layer, dim] native features (cast to STORE_DTYPE) plus its
        index row. Temp-write + fsync + rename, so a crash can never leave a short file behind a
        'done' row; the caller commits."""
        arr = np.ascontiguousarray(feat, dtype=STORE_DTYPE)
        n_frame, n_layer, dim = arr.shape
        str_rel = str_audio_key + FEATURE_FILE_SUFFIX
        path_out = self._path_features / str_rel
        path_tmp = path_out.with_name(path_out.name + ".tmp")
        with open(path_tmp, "wb") as handle:
            handle.write(arr.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        path_tmp.replace(path_out)  # atomic within the same directory
        self._conn.execute("INSERT OR REPLACE INTO audio_feature VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (str_audio_key, music_id, n_frame, n_layer, dim, str_rel,
                            msgpack.packb(vec_anchor), end_beat))

    def read_feature(self, str_audio_key: str, frame_lo: int = None, frame_hi: int = None) -> np.ndarray:
        """Read [n_frame, n_layer, dim] native features, or just frames [frame_lo, frame_hi) x all
        layers in ONE seek+read (frame-major = one contiguous byte range). Bounds are clamped.
        buffering=0 gives a raw FileIO (one syscall into the result bytes); per-call handles keep
        DataLoader workers free of shared file-position hazards."""
        row = self._conn.execute("SELECT n_frame, n_layer, dim, feat_rel FROM audio_feature WHERE audio_key = ?",
                                 (str_audio_key,)).fetchone()
        if row is None:
            raise KeyError(str_audio_key)
        n_frame, n_layer, dim, str_rel = row
        bytes_per_frame = n_layer * dim * np.dtype(STORE_DTYPE).itemsize
        index_lo = 0 if frame_lo is None else max(0, frame_lo)
        index_hi = max(index_lo, n_frame if frame_hi is None else min(n_frame, frame_hi))
        n_want = (index_hi - index_lo) * bytes_per_frame
        with open(self._path_features / str_rel, "rb", buffering=0) as handle:
            handle.seek(index_lo * bytes_per_frame)
            raw = handle.read(n_want)
        if len(raw) != n_want:  # truncated file: fail loudly, never silently short
            raise OSError(f"{str_rel}: read {len(raw)} of {n_want} bytes at frame {index_lo}")
        return np.frombuffer(raw, dtype=STORE_DTYPE).reshape(index_hi - index_lo, n_layer, dim)

    def read_timing(self, str_audio_key: str) -> tuple[list, int]:
        """(bpm anchors [[beat, ms, bpm], ...], end_beat) for a song"""
        row = self._conn.execute("SELECT timing_blob, end_beat FROM audio_feature WHERE audio_key = ?",
                                 (str_audio_key,)).fetchone()
        if row is None:
            raise KeyError(str_audio_key)
        return msgpack.unpackb(row[0], raw=False), row[1]

    def iter_audio_keys(self) -> list:
        """(audio_key, music_id, n_frame) for every stored song"""
        return list(self._conn.execute("SELECT audio_key, music_id, n_frame FROM audio_feature ORDER BY audio_key"))
