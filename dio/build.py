"""manic.sqlite builder: .cache/seq (extracted d*.sq3 + spu*d.va3) + mdb_fz.xml [+ music root]
-> one denormalised row per (song x difficulty) drum chart.

Usage:
  python -m dio.build --cache <dir> --mdb <mdb_fz.xml> --out <manic.sqlite> [--music-root <dir>]

Cache dirs are named m{seq_id:04} (the extractor's convention); seq_id parsed from the name is
cross-checked against nothing, but the sq3-embedded music_id is cross-checked against mdb's."""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from msql import write_charts
from msql.blob import pack_ir

from .audio_index import build_title_index
from .common_struct import SongIr
from .instrument import CLASS_UNNAMED, classify_name, is_exotic
from .ir_grid import chart_to_grid
from .mdb import difnum_of, parse_mdb
from .sq3 import parse_drum_sq3
from .va3 import parse_keysound_table

_LANE_BASS = 2


@dataclass
class BuildStats:
    cnt_song: int = 0
    cnt_chart: int = 0
    cnt_failed: int = 0
    cnt_no_mdb: int = 0
    cnt_no_audio: int = 0
    cnt_music_id_mismatch: int = 0


def _song_rows(dir_song: Path, map_seq_record: dict, map_title_relpath: dict, stats: BuildStats):
    seq_id = int(dir_song.name.removeprefix("m"))
    vec_path_sq3 = sorted(dir_song.glob("d*.sq3"))
    vec_path_va3 = sorted(dir_song.glob("spu*d.va3"))
    if not vec_path_sq3 or not vec_path_va3:
        raise FileNotFoundError(f"missing members: sq3={len(vec_path_sq3)} va3={len(vec_path_va3)}")

    music_id, timing, vec_chart = parse_drum_sq3(vec_path_sq3[0].read_bytes())
    vec_keysound = parse_keysound_table(vec_path_va3[0].read_bytes())
    song = SongIr(music_id=music_id, seq_id=seq_id, timing=timing, vec_keysound=vec_keysound, vec_chart=vec_chart)

    map_sid_name = {entry.sound_id: entry.name for entry in vec_keysound}
    set_sid_kick_lane = {note.sound_id for chart in vec_chart for note in chart.vec_note
                         if note.lane == _LANE_BASS and not note.is_auto}

    # song-level instrumentation labels over every chart's playable notes (m1f0s4 semantics)
    cnt_playable, cnt_exotic, cnt_unnamed = 0, 0, 0
    for chart in vec_chart:
        for note in chart.vec_note:
            if note.is_auto:
                continue
            cnt_playable += 1
            name_class = classify_name(map_sid_name.get(note.sound_id, ""))
            cnt_exotic += is_exotic(name_class)
            cnt_unnamed += name_class == CLASS_UNNAMED
    exotic_share = cnt_exotic / cnt_playable if cnt_playable else 0.0
    unnamed_share = cnt_unnamed / cnt_playable if cnt_playable else 0.0

    record = map_seq_record.get(seq_id)
    if record is None:
        stats.cnt_no_mdb += 1
    elif record.music_id != music_id:
        stats.cnt_music_id_mismatch += 1
    title = record.title if record else ""
    audio_path = map_title_relpath.get(title, "")
    if not audio_path:
        stats.cnt_no_audio += 1

    for chart in vec_chart:
        grid = chart_to_grid(chart, map_sid_name, set_sid_kick_lane)
        vec_playable = [note for note in chart.vec_note if not note.is_auto]
        yield (
            f"{dir_song.name}_d{chart.difficulty}", music_id, seq_id, chart.difficulty,
            difnum_of(record, chart.difficulty) if record else 0,
            title, record.bpm if record else 0, record.bpm2 if record else 0,
            len(chart.vec_note), len(vec_playable), grid["n_offgrid"],
            exotic_share, unnamed_share,
            audio_path, pack_ir(song.to_chart_dict(chart)), pack_ir(grid),
        )
        stats.cnt_chart += 1
    stats.cnt_song += 1


def build_library(path_cache, path_mdb, path_out, path_music_root=None) -> BuildStats:
    map_seq_record = parse_mdb(path_mdb)
    map_title_relpath = {}
    if path_music_root:
        map_title_relpath, cnt_collision = build_title_index(path_music_root)
        print(f"audio index: {len(map_title_relpath)} titles ({cnt_collision} collisions dropped)")

    stats = BuildStats()

    def rows():
        for dir_song in sorted(path for path in Path(path_cache).iterdir() if path.is_dir()):
            try:
                yield from _song_rows(dir_song, map_seq_record, map_title_relpath, stats)
            except Exception as error:
                stats.cnt_failed += 1
                print(f"FAIL {dir_song.name}: {error}")

    write_charts(path_out, rows())
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="extracted seq cache dir (m1f0_extract_seq_cache.py output)")
    parser.add_argument("--mdb", required=True, help="mdb_fz.xml path")
    parser.add_argument("--out", required=True, help="manic.sqlite output path")
    parser.add_argument("--music-root", default=None, help="GitadoraOnEar opus library root (optional)")
    args = parser.parse_args()

    stats = build_library(args.cache, args.mdb, args.out, args.music_root)
    print(f"songs={stats.cnt_song} charts={stats.cnt_chart} failed={stats.cnt_failed} "
          f"no_mdb={stats.cnt_no_mdb} no_audio={stats.cnt_no_audio} "
          f"music_id_mismatch={stats.cnt_music_id_mismatch}")
    return 0 if stats.cnt_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
