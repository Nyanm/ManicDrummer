"""Evaluation for the detection baseline (f4s3): per-lane event-F1 at a wall-clock tolerance,
velocity MAE on matched events, lane-8 pedal semantic accuracy, and the simultaneity legality rate.

Events on both sides live on the tatum grid; matching converts tatums to milliseconds through the
song's authoritative BeatTimeMap and greedily one-to-one matches within +-TOLERANCE_MS per lane
(the ADT-standard 50ms window). Per-song measurement, aggregated micro (event-weighted) --
the paired per-song records are kept so later runs can Wilcoxon against this baseline."""
from dataclasses import dataclass, field

import numpy as np

from dio.common_struct import N_LANE, PEDAL_BD, PEDAL_HH

TOLERANCE_MS = 50.0
MAX_SIMULTANEOUS = 2


@dataclass
class LaneTally:
    cnt_match: int = 0
    cnt_pred: int = 0
    cnt_true: int = 0
    vec_velocity_error: list = field(default_factory=list)

    def precision(self) -> float:
        return self.cnt_match / self.cnt_pred if self.cnt_pred else 0.0

    def recall(self) -> float:
        return self.cnt_match / self.cnt_true if self.cnt_true else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if p + r else 0.0


def match_events(arr_true_ms: np.ndarray, arr_pred_ms: np.ndarray) -> list:
    """Greedy one-to-one matching within TOLERANCE_MS on sorted event times; returns index pairs"""
    vec_pair = []
    index_pred = 0
    for index_true, time_true in enumerate(arr_true_ms):
        while index_pred < len(arr_pred_ms) and arr_pred_ms[index_pred] < time_true - TOLERANCE_MS:
            index_pred += 1
        if index_pred < len(arr_pred_ms) and abs(arr_pred_ms[index_pred] - time_true) <= TOLERANCE_MS:
            vec_pair.append((index_true, index_pred))
            index_pred += 1
    return vec_pair


@dataclass
class SongEval:
    audio_key: str
    vec_lane: list                    # 9 LaneTally
    cnt_pedal_match: int = 0          # matched lane-8 events with a known true semantic
    cnt_pedal_correct: int = 0
    cnt_tatum_illegal: int = 0        # predicted tatums with > MAX_SIMULTANEOUS onsets
    cnt_tatum_pred: int = 0


def evaluate_song(audio_key: str, time_map, arr_true_onset: np.ndarray, arr_true_velocity: np.ndarray,
                  arr_true_pedal: np.ndarray, arr_pred_onset: np.ndarray, arr_pred_velocity: np.ndarray,
                  arr_pred_pedal: np.ndarray, tatum_per_beat: int) -> SongEval:
    """All arrays are [n_tatum, ...] on the same grid; pred arrays are already thresholded/argmaxed"""
    unit_per_tatum = 480 // tatum_per_beat
    result = SongEval(audio_key=audio_key, vec_lane=[LaneTally() for _ in range(N_LANE)])
    n_tatum = min(len(arr_true_onset), len(arr_pred_onset))

    for lane in range(N_LANE):
        vec_true = np.flatnonzero(arr_true_onset[:n_tatum, lane])
        vec_pred = np.flatnonzero(arr_pred_onset[:n_tatum, lane])
        arr_true_ms = np.asarray(time_map.ms_of_beat(vec_true * unit_per_tatum))
        arr_pred_ms = np.asarray(time_map.ms_of_beat(vec_pred * unit_per_tatum))
        vec_pair = match_events(arr_true_ms, arr_pred_ms)
        tally = result.vec_lane[lane]
        tally.cnt_match += len(vec_pair)
        tally.cnt_pred += len(vec_pred)
        tally.cnt_true += len(vec_true)
        for index_true, index_pred in vec_pair:
            tally.vec_velocity_error.append(abs(float(arr_true_velocity[vec_true[index_true], lane])
                                                - float(arr_pred_velocity[vec_pred[index_pred], lane])))
            if lane == 8:
                pedal_true = int(arr_true_pedal[vec_true[index_true]])
                if pedal_true in (PEDAL_HH, PEDAL_BD):
                    result.cnt_pedal_match += 1
                    result.cnt_pedal_correct += int(arr_pred_pedal[vec_pred[index_pred]]) == pedal_true

    arr_count = arr_pred_onset[:n_tatum].sum(axis=1)
    result.cnt_tatum_pred = int((arr_count > 0).sum())
    result.cnt_tatum_illegal = int((arr_count > MAX_SIMULTANEOUS).sum())
    return result


def report(vec_song_eval: list) -> dict:
    """Aggregate micro numbers + per-lane table; returns the dict it prints (for logging)"""
    total = [LaneTally() for _ in range(N_LANE)]
    cnt_pedal_match = cnt_pedal_correct = cnt_illegal = cnt_tatum = 0
    for song in vec_song_eval:
        for lane in range(N_LANE):
            total[lane].cnt_match += song.vec_lane[lane].cnt_match
            total[lane].cnt_pred += song.vec_lane[lane].cnt_pred
            total[lane].cnt_true += song.vec_lane[lane].cnt_true
            total[lane].vec_velocity_error.extend(song.vec_lane[lane].vec_velocity_error)
        cnt_pedal_match += song.cnt_pedal_match
        cnt_pedal_correct += song.cnt_pedal_correct
        cnt_illegal += song.cnt_tatum_illegal
        cnt_tatum += song.cnt_tatum_pred

    micro = LaneTally(cnt_match=sum(t.cnt_match for t in total), cnt_pred=sum(t.cnt_pred for t in total),
                      cnt_true=sum(t.cnt_true for t in total))
    vec_velocity_all = [err for t in total for err in t.vec_velocity_error]
    summary = {
        "micro_f1": micro.f1(), "micro_p": micro.precision(), "micro_r": micro.recall(),
        "lane_f1": [t.f1() for t in total],
        "velocity_mae": float(np.mean(vec_velocity_all)) if vec_velocity_all else None,
        "pedal_acc": cnt_pedal_correct / cnt_pedal_match if cnt_pedal_match else None,
        "illegal_rate": cnt_illegal / cnt_tatum if cnt_tatum else 0.0,
        "n_song": len(vec_song_eval),
    }
    lane_names = ["HH", "SD", "BD", "HT", "LT", "RC", "LC", "FT", "LP"]
    print(f"event-F1 micro {summary['micro_f1']:.3f} (P {summary['micro_p']:.3f} R {summary['micro_r']:.3f}) "
          f"over {summary['n_song']} songs")
    print("  per lane: " + "  ".join(f"{name} {t.f1():.3f}" for name, t in zip(lane_names, total)))
    print(f"  velocity MAE {summary['velocity_mae']:.4f}  pedal acc "
          f"{summary['pedal_acc'] if summary['pedal_acc'] is not None else float('nan'):.3f}  "
          f"illegal>{MAX_SIMULTANEOUS} rate {summary['illegal_rate']:.4f}")
    return summary
