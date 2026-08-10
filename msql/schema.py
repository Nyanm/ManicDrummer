"""The `charts` table schema of manic.sqlite -- one denormalised row per (song x difficulty) drum chart.
Single definition; the writer (dio.build) and every reader reference COLUMNS from here.

Keys: chart_id = "m{seq_id:04}_d{difficulty}". music_id / seq_id are the join keys to game data;
audio_path is a locator RELATIVE to the music root ("" when unresolved) and never a join key."""

COLUMNS = [
    "chart_id", "music_id", "seq_id", "difficulty", "difnum",
    "title", "bpm", "bpm2",
    "n_note", "n_playable", "n_offgrid",
    "exotic_share", "unnamed_share",
    "audio_path", "ir_rich", "ir_grid",
]

SCHEMA = """
CREATE TABLE charts (
    chart_id      TEXT PRIMARY KEY,
    music_id      INTEGER NOT NULL,
    seq_id        INTEGER NOT NULL,
    difficulty    INTEGER NOT NULL,
    difnum        INTEGER NOT NULL,
    title         TEXT NOT NULL,
    bpm           INTEGER NOT NULL,
    bpm2          INTEGER NOT NULL,
    n_note        INTEGER NOT NULL,
    n_playable    INTEGER NOT NULL,
    n_offgrid     INTEGER NOT NULL,
    exotic_share  REAL NOT NULL,
    unnamed_share REAL NOT NULL,
    audio_path    TEXT NOT NULL,
    ir_rich       BLOB NOT NULL,
    ir_grid       BLOB NOT NULL
)
"""
