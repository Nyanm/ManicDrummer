"""Beat-grid alignment: interpolate native 25Hz MuQ frames onto the authored beat grid at read time.

The chart side proved (m1f1s0) that every song carries an authoritative piecewise-constant-BPM
timing map whose anchors are exact to one 1/300s tick. This module turns that map into wall-clock
times for beat-grid feature frames (GRID_STEPS_PER_BEAT per beat) and linearly interpolates the
native frames there. The cache asset stays timing-independent; grid resolution is a free
hyperparameter (re-read, never re-encode) -- the survey's "beat-aligned grid as pure CPU
post-processing" judgment, applied from day one. No torch here."""
import numpy as np

from dio.common_struct import BEAT_DIVISION

from .. import config


class BeatTimeMap:
    """Piecewise-linear beat -> milliseconds from bpm anchors [[beat, ms, bpm], ...] (beat in native
    1/480 units, beat-ascending, first anchor at beat 0). Anchor ms comes from the anchor's own tick
    (authoritative wall clock); between anchors time advances at the anchor's bpm."""

    def __init__(self, vec_anchor: list):
        if not vec_anchor or vec_anchor[0][0] != 0:
            raise ValueError(f"bpm anchors must start at beat 0, got {vec_anchor[:1]}")
        arr = np.asarray(vec_anchor, dtype=np.float64)
        self.arr_beat = arr[:, 0]
        self.arr_ms = arr[:, 1]
        self.arr_ms_per_unit = 60000.0 / arr[:, 2] / BEAT_DIVISION  # ms per 1/480-beat unit, per segment

    @classmethod
    def from_timing_dict(cls, timing: dict) -> "BeatTimeMap":
        """From an ir_rich blob's `timing` section (vec_bpm rows are [tick, beat, bpm])"""
        return cls([[beat, tick / timing["time_division"] * 1000.0, bpm]
                    for tick, beat, bpm in timing["vec_bpm"]])

    def ms_of_beat(self, arr_beat_unit) -> np.ndarray:
        """Wall-clock ms for positions given in native 1/480-beat units (scalar or array)"""
        arr_beat_unit = np.asarray(arr_beat_unit, dtype=np.float64)
        index_segment = np.clip(np.searchsorted(self.arr_beat, arr_beat_unit, side="right") - 1, 0, None)
        return self.arr_ms[index_segment] \
            + (arr_beat_unit - self.arr_beat[index_segment]) * self.arr_ms_per_unit[index_segment]

    def beat_of_ms(self, time_ms: float) -> float:
        """Inverse map: the (fractional) 1/480-beat position at a wall-clock time. Charts can
        outlive their audio (endpos past the opus end), so consumers use this to cap sampling."""
        index_segment = max(0, int(np.searchsorted(self.arr_ms, time_ms, side="right")) - 1)
        return float(self.arr_beat[index_segment]
                     + (time_ms - self.arr_ms[index_segment]) / self.arr_ms_per_unit[index_segment])


def grid_frame_beats(end_beat: int, steps_per_beat: int = config.GRID_STEPS_PER_BEAT) -> np.ndarray:
    """The beat positions (native 1/480 units) of every grid feature frame from beat 0 to end_beat"""
    stride_unit = BEAT_DIVISION // steps_per_beat
    return np.arange(0, end_beat + stride_unit, stride_unit, dtype=np.int64)


def resample_to_grid(feat_native: np.ndarray, arr_time_ms: np.ndarray) -> np.ndarray:
    """Linearly interpolate native frame-major features [n_frame, n_layer, dim] at the given wall
    times -> [len(arr_time_ms), n_layer, dim]. Positions are clamped to the valid frame range, so a
    grid frame past the audio end holds the last native frame rather than garbage."""
    if len(feat_native) == 0:
        raise ValueError("resample_to_grid got an empty feature slice -- window sampled past the audio end?")
    arr_pos = np.asarray(arr_time_ms, dtype=np.float64) / 1000.0 * config.MUQ_FRAME_HZ
    arr_pos = np.clip(arr_pos, 0.0, len(feat_native) - 1.0)
    index_lo = arr_pos.astype(np.int64)
    index_hi = np.minimum(index_lo + 1, len(feat_native) - 1)
    frac = (arr_pos - index_lo).astype(feat_native.dtype if feat_native.dtype.kind == "f" else np.float32)
    lo = feat_native[index_lo].astype(np.float32)
    hi = feat_native[index_hi].astype(np.float32)
    return lo + (hi - lo) * frac[:, None, None].astype(np.float32)


def grid_features(feat_native: np.ndarray, time_map: BeatTimeMap, end_beat: int,
                  steps_per_beat: int = config.GRID_STEPS_PER_BEAT) -> np.ndarray:
    """The one-call form: native features + timing -> beat-grid features [n_grid_frame, n_layer, dim]"""
    return resample_to_grid(feat_native, time_map.ms_of_beat(grid_frame_beats(end_beat, steps_per_beat)))
