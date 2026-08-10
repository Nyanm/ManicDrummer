"""Feature-build orchestration: the resumable MuQ cache build over manic.sqlite's songs.

For each song with a resolved audio_path: load its opus (mono 24kHz) -> MuQ encode -> write the
native [n_frame, 13, 1024] fp16 feature file + index row (bpm anchors + end_beat from the chart's
authoritative timing) -- ONE commit per song, so an overnight crash never loses completed work.
Already-done songs are skipped; a failed song records its error and the run continues.

Usage (the big GPU run lives on lab):
  python -m md.encode.build --manic <manic.sqlite> --music-root <opus root> --audio-db <manic_audio.sqlite>
"""
import argparse
import sys
import time
from pathlib import Path

from msql import connect_ro, read_ir_rich
from msql.audio_db import AudioDb, FEATURE_LAYOUT

from .. import config
from . import audio_io, muq_runner


def _iter_songs(conn) -> list:
    """(audio_key, music_id, chart_id, audio_path) per unique song with resolved audio; the chart_id
    is any one row of the song (timing is identical across difficulties)."""
    return [(f"m{seq_id:04d}", music_id, chart_id, audio_path) for seq_id, music_id, chart_id, audio_path in
            conn.execute("SELECT seq_id, music_id, MIN(chart_id), audio_path FROM charts "
                         "WHERE audio_path != '' GROUP BY seq_id ORDER BY seq_id")]


def _timing_of(conn, chart_id: str) -> tuple[list, int]:
    """(bpm anchors [[beat, ms, bpm], ...], end_beat) from a chart row's authoritative timing"""
    timing = read_ir_rich(conn, chart_id)["timing"]
    vec_anchor = [[beat, tick / timing["time_division"] * 1000.0, bpm] for tick, beat, bpm in timing["vec_bpm"]]
    return vec_anchor, timing["end_beat"]


def _record_meta(db: AudioDb) -> None:
    """Stamp the model/grid constants into meta so the cache is self-describing"""
    db.set_meta("muq_model", config.MUQ_MODEL_NAME)
    db.set_meta("frame_hz", config.MUQ_FRAME_HZ)
    db.set_meta("dim", config.MUQ_DIM)
    db.set_meta("n_layer", config.MUQ_N_LAYER)
    db.set_meta("store_dtype", config.STORE_DTYPE)
    db.set_meta("layout", FEATURE_LAYOUT)
    db.commit()


def run_build(path_manic, path_music_root, path_audio_db, str_device: str = "cuda", limit: int = 0) -> dict:
    conn = connect_ro(path_manic)
    db = AudioDb.open_write(path_audio_db)
    _record_meta(db)

    vec_song = _iter_songs(conn)
    if limit > 0:
        vec_song = vec_song[:limit]
    cnt_total = len(vec_song)
    print(f"building MuQ features for {cnt_total} songs on {str_device} -> {path_audio_db}", file=sys.stderr)

    cnt_done = cnt_skip = cnt_fail = 0
    for index_song, (audio_key, music_id, chart_id, audio_path) in enumerate(vec_song, start=1):
        if db.is_done(audio_key):
            cnt_skip += 1
            continue
        time_begin = time.perf_counter()
        try:
            path_opus = Path(path_music_root) / audio_path
            if not path_opus.exists():
                raise FileNotFoundError(f"opus not found: {path_opus}")
            vec_anchor, end_beat = _timing_of(conn, chart_id)
            wav = audio_io.load_mono(str(path_opus))
            feat = muq_runner.encode(wav, str_device)  # [n_frame, 13, 1024] fp32
            db.write_feature(audio_key, music_id, feat, vec_anchor, end_beat)
            db.mark_done(audio_key, time.perf_counter() - time_begin)
            db.commit()  # one durable commit per song
            cnt_done += 1
            print(f"[{index_song}/{cnt_total}] {audio_key} ok n_frame={feat.shape[0]} "
                  f"{time.perf_counter() - time_begin:.1f}s", file=sys.stderr)
        except Exception as exc:  # one bad song must not abort the run
            db.mark_failed(audio_key, f"{type(exc).__name__}: {exc}")
            db.commit()
            cnt_fail += 1
            print(f"[{index_song}/{cnt_total}] {audio_key} FAIL {type(exc).__name__}: {exc}", file=sys.stderr)

    db.close()
    conn.close()
    print(f"done={cnt_done} skipped={cnt_skip} failed={cnt_fail} total={cnt_total}", file=sys.stderr)
    return {"total": cnt_total, "done": cnt_done, "skipped": cnt_skip, "failed": cnt_fail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manic", required=True, help="manic.sqlite path")
    parser.add_argument("--music-root", required=True, help="opus library root (audio_path is relative to it)")
    parser.add_argument("--audio-db", required=True, help="manic_audio.sqlite output path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="build only the first N songs (probe runs)")
    args = parser.parse_args()
    stats = run_build(args.manic, args.music_root, args.audio_db, args.device, args.limit)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
