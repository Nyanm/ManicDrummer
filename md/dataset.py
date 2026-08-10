"""Window dataset for the detection-head baseline: beat-grid MuQ features x tatum-level drum labels.

One sample = a window of WINDOW_BEATS beats from one song:
  features [n_tatum, 13, 1024]  read-time beat alignment (md.encode.beat_grid) at TATUM_PER_BEAT
  onset    [n_tatum, 9]         multi-hot lanes (the song's FULL playable transcription, see below)
  velocity [n_tatum, 9]         0..1 (velocity/127) where onset, else 0
  pedal    [n_tatum]            PEDAL_NONE / PEDAL_HH / PEDAL_BD / PEDAL_UNKNOWN at lane-8 onsets

Targets are the song-level union of every difficulty's PLAYABLE notes (m1f0s3: difficulties
re-select playable/auto over one shared transcription, so the union IS the maximal chart;
auto-only sounds carry no lane and are excluded -- they are audio, not chart). Tatum = 1/24 beat:
resolves 16ths/24ths/32nds/48ths exactly (odd-1/48 events round down, ~1% of notes are finer
anyway). Split is by SONG (every difficulty of a song shares its windows by construction).
"""
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from dio.common_struct import GRID_PER_BEAT, N_LANE, PEDAL_NONE
from md.encode.beat_grid import BeatTimeMap, grid_frame_beats, resample_to_grid
from msql import connect_ro, read_ir_grid, read_ir_rich
from msql.audio_db import AudioDb

TATUM_PER_BEAT = 24                          # 1/24 beat; GRID_PER_BEAT // TATUM_PER_BEAT = 2 grid ticks
_GRID_PER_TATUM = GRID_PER_BEAT // TATUM_PER_BEAT
WINDOW_BEATS = 16
WINDOW_TATUMS = WINDOW_BEATS * TATUM_PER_BEAT
VAL_STRIDE = 57                              # every 57th song (seq_id order) is validation, ~24 songs


@dataclass
class SongEntry:
    audio_key: str                           # "m{seq_id:04}"
    end_beat: int                            # native 1/480 units; min(chart endpos, audio end) -- charts
                                             # can outlive their opus, windows must not sample past audio
    n_frame: int                             # native 25Hz frames in the cache
    time_map: BeatTimeMap
    arr_onset: np.ndarray                    # [n_tatum, 9] uint8
    arr_velocity: np.ndarray                 # [n_tatum, 9] float32, 0..1
    arr_pedal: np.ndarray                    # [n_tatum] uint8


def load_song_entries(path_manic, path_audio_db) -> list:
    """Build every song's tatum label arrays from manic.sqlite (union of playable rows across
    difficulties, max velocity on collision) joined with the feature cache's timing."""
    conn = connect_ro(path_manic)
    db = AudioDb.open_read(path_audio_db)
    map_key_frames = {audio_key: n_frame for audio_key, _, n_frame in db.iter_audio_keys()}

    vec_entry = []
    vec_seq = [row[0] for row in conn.execute(
        "SELECT DISTINCT seq_id FROM charts WHERE audio_path != '' ORDER BY seq_id")]
    for seq_id in vec_seq:
        audio_key = f"m{seq_id:04d}"
        if audio_key not in map_key_frames:
            continue
        vec_anchor, end_beat = db.read_timing(audio_key)
        time_map = BeatTimeMap(vec_anchor)
        n_frame = map_key_frames[audio_key]
        # charts can outlive their audio: cap the usable range at the audio's last full frame
        end_beat = min(end_beat, int(time_map.beat_of_ms((n_frame - 1) * 40.0)))
        n_tatum = end_beat // (480 // TATUM_PER_BEAT) + 1
        arr_onset = np.zeros((n_tatum, N_LANE), dtype=np.uint8)
        arr_velocity = np.zeros((n_tatum, N_LANE), dtype=np.float32)
        arr_pedal = np.full(n_tatum, PEDAL_NONE, dtype=np.uint8)
        for chart_id, in conn.execute("SELECT chart_id FROM charts WHERE seq_id = ?", (seq_id,)):
            for grid48, _residue, lane, velocity, pedal, playable in read_ir_grid(conn, chart_id)["vec_note"]:
                if not playable or lane >= N_LANE:
                    continue
                tatum = grid48 // _GRID_PER_TATUM
                if tatum >= n_tatum:  # event past the audio end (chart outlives opus): drop, never clamp
                    continue
                arr_onset[tatum, lane] = 1
                arr_velocity[tatum, lane] = max(arr_velocity[tatum, lane], velocity / 127.0)
                if lane == 8 and pedal != PEDAL_NONE:
                    arr_pedal[tatum] = pedal
        vec_entry.append(SongEntry(audio_key=audio_key, end_beat=end_beat, n_frame=n_frame,
                                   time_map=time_map,
                                   arr_onset=arr_onset, arr_velocity=arr_velocity, arr_pedal=arr_pedal))
    conn.close()
    db.close()
    return vec_entry


def split_entries(vec_entry: list) -> tuple[list, list]:
    """Deterministic song-level split: every VAL_STRIDE-th song (seq order) validates"""
    vec_val = vec_entry[::VAL_STRIDE]
    set_val = {entry.audio_key for entry in vec_val}
    return [entry for entry in vec_entry if entry.audio_key not in set_val], vec_val


class WindowDataset(Dataset):
    """Random (train) or tiled (val) windows over songs. Feature reads open the AudioDb lazily per
    worker (sqlite connections cannot cross process boundaries)."""

    def __init__(self, vec_entry: list, path_audio_db, windows_per_song: int = 4, tiled: bool = False):
        self.vec_entry = vec_entry
        self.path_audio_db = path_audio_db
        self.windows_per_song = windows_per_song
        self.tiled = tiled
        self._db = None
        self.vec_index = []                  # (index_entry, beat_start) -- beat_start in native 1/480 units
        window_units = WINDOW_BEATS * 480
        for index_entry, entry in enumerate(vec_entry):
            if self.tiled:
                for beat_start in range(0, max(1, entry.end_beat - window_units), window_units):
                    self.vec_index.append((index_entry, beat_start))
            else:
                self.vec_index.extend((index_entry, -1) for _ in range(windows_per_song))

    def __len__(self) -> int:
        return len(self.vec_index)

    def _database(self) -> AudioDb:
        if self._db is None:
            self._db = AudioDb.open_read(self.path_audio_db)
        return self._db

    def __getitem__(self, index: int) -> dict:
        index_entry, beat_start = self.vec_index[index]
        entry = self.vec_entry[index_entry]
        window_units = WINDOW_BEATS * 480
        if beat_start < 0:  # train: random window (beat-aligned start keeps the phase embedding honest)
            beat_max = max(0, entry.end_beat - window_units)
            beat_start = random.randrange(0, beat_max + 480, 480) if beat_max > 0 else 0

        # features: wall times of the window's tatums -> native frame range -> interpolate
        arr_beat = grid_frame_beats(window_units - 480 // TATUM_PER_BEAT, TATUM_PER_BEAT) + beat_start
        arr_time_ms = entry.time_map.ms_of_beat(arr_beat)
        frame_lo = min(max(0, int(arr_time_ms[0] / 40.0) - 2), max(0, entry.n_frame - 2))
        frame_hi = min(int(arr_time_ms[-1] / 40.0) + 3, entry.n_frame)
        feat_native = self._database().read_feature(entry.audio_key, frame_lo, frame_hi)
        feat = resample_to_grid(feat_native, arr_time_ms - frame_lo * 40.0)

        tatum_lo = beat_start // (480 // TATUM_PER_BEAT)
        pad = WINDOW_TATUMS - min(WINDOW_TATUMS, len(entry.arr_onset) - tatum_lo)
        take = WINDOW_TATUMS - pad
        onset = np.zeros((WINDOW_TATUMS, N_LANE), dtype=np.float32)
        velocity = np.zeros((WINDOW_TATUMS, N_LANE), dtype=np.float32)
        pedal = np.full(WINDOW_TATUMS, PEDAL_NONE, dtype=np.int64)
        onset[:take] = entry.arr_onset[tatum_lo:tatum_lo + take]
        velocity[:take] = entry.arr_velocity[tatum_lo:tatum_lo + take]
        pedal[:take] = entry.arr_pedal[tatum_lo:tatum_lo + take]

        return {
            "feat": torch.from_numpy(np.ascontiguousarray(feat, dtype=np.float32)),
            "onset": torch.from_numpy(onset),
            "velocity": torch.from_numpy(velocity),
            "pedal": torch.from_numpy(pedal),
            "index_entry": index_entry,
            "tatum_lo": tatum_lo,
        }
