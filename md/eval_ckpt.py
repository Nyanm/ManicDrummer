"""Offline checkpoint evaluation (M2 f0): baseline re-measurement, per-lane threshold calibration
and the top-2 simultaneity constraint -- all as decode-time replay over one inference pass.

Threshold tuning is 2-fold cross-fitted to stay honest: val songs split even/odd, each fold is
decoded with thresholds tuned on the OTHER fold, then all songs aggregate. Tuning maximises
per-lane event-F1 (event matching at +-50ms, the same metric that is reported).

Per-song records are written as JSON for md-side paired run comparison (f0s4 Wilcoxon gate).

Usage (on lab, where the feature cache lives):
  python -m md.eval_ckpt --ckpt <pt> --manic <sqlite> --audio-db <sqlite>
                         [--sweep] [--top2] [--records out.json]
"""
import argparse
import json
import sys

import numpy as np
import torch

from dio.common_struct import N_LANE

from .dataset import TATUM_PER_BEAT, load_song_entries, split_entries
from .detector import DrumDetector
from .evaluate import evaluate_song, match_events, report
from .infer import decode_onsets, infer_song_probs

THRESHOLD_GRID = [round(0.15 + 0.05 * index, 2) for index in range(15)]  # 0.15 .. 0.85


def lane_f1_at(entry, arr_prob: np.ndarray, lane: int, threshold: float) -> float:
    """One lane's event-F1 at a candidate threshold (matching in wall-clock via the song's map)"""
    unit_per_tatum = 480 // TATUM_PER_BEAT
    vec_true = np.flatnonzero(entry.arr_onset[:, lane])
    vec_pred = np.flatnonzero(arr_prob[:, lane] >= threshold)
    if len(vec_true) == 0 and len(vec_pred) == 0:
        return 1.0
    if len(vec_true) == 0 or len(vec_pred) == 0:
        return 0.0
    cnt_match = len(match_events(np.asarray(entry.time_map.ms_of_beat(vec_true * unit_per_tatum)),
                                 np.asarray(entry.time_map.ms_of_beat(vec_pred * unit_per_tatum))))
    precision, recall = cnt_match / len(vec_pred), cnt_match / len(vec_true)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def tune_thresholds(vec_entry: list, map_result: dict, vec_index: list) -> list:
    """Per-lane thresholds maximising mean event-F1 over the tuning songs"""
    vec_threshold = []
    for lane in range(N_LANE):
        best = max(THRESHOLD_GRID,
                   key=lambda threshold: float(np.mean([lane_f1_at(vec_entry[index], map_result[index]["onset_prob"],
                                                                   lane, threshold) for index in vec_index])))
        vec_threshold.append(best)
    return vec_threshold


def evaluate_fold(vec_entry, map_result, vec_index, vec_threshold, top2: bool) -> list:
    vec_song_eval = []
    for index in vec_index:
        entry, result = vec_entry[index], map_result[index]
        arr_onset = decode_onsets(result["onset_prob"], vec_threshold, top2=top2)
        vec_song_eval.append(evaluate_song(entry.audio_key, entry.time_map, entry.arr_onset,
                                           entry.arr_velocity, entry.arr_pedal, arr_onset,
                                           result["velocity"], result["pedal"], TATUM_PER_BEAT))
    return vec_song_eval


def dump_records(path_out: str, protocol: dict, vec_song_eval: list) -> None:
    vec_song = [{"audio_key": song.audio_key,
                 "lane": [[t.cnt_match, t.cnt_pred, t.cnt_true] for t in song.vec_lane],
                 "pedal": [song.cnt_pedal_correct, song.cnt_pedal_match],
                 "illegal": [song.cnt_tatum_illegal, song.cnt_tatum_pred]}
                for song in vec_song_eval]
    with open(path_out, "w", encoding="utf-8") as handle:
        json.dump({"protocol": protocol, "songs": vec_song}, handle, indent=1)
    print(f"records -> {path_out}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manic", required=True)
    parser.add_argument("--audio-db", required=True)
    parser.add_argument("--sweep", action="store_true", help="2-fold cross-fitted per-lane thresholds")
    parser.add_argument("--top2", action="store_true", help="apply the simultaneity constraint")
    parser.add_argument("--records", default="", help="write per-song records JSON here")
    parser.add_argument("--save-calibration", action="store_true",
                        help="tune per-lane thresholds on ALL val songs and write <ckpt>.calibration.json "
                             "(the production decode artifact; honest eval numbers still come from --sweep)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _, vec_val = split_entries(load_song_entries(args.manic, args.audio_db))
    print(f"val songs: {len(vec_val)}", file=sys.stderr)
    state = torch.load(args.ckpt, map_location=args.device)
    model = DrumDetector(use_mel=state.get("use_mel", False)).to(args.device)
    model.load_state_dict(state["model"])
    map_result = infer_song_probs(model, vec_val, args.audio_db, args.device, args.batch,
                                  state.get("window_beats", 16))

    protocol = {"ckpt": args.ckpt, "sweep": args.sweep, "top2": args.top2, "n_song": len(vec_val)}
    if not args.sweep:
        vec_song_eval = evaluate_fold(vec_val, map_result, list(range(len(vec_val))), [0.5] * N_LANE, args.top2)
        protocol["thresholds"] = [0.5] * N_LANE
    else:
        fold_a, fold_b = list(range(0, len(vec_val), 2)), list(range(1, len(vec_val), 2))
        threshold_for_b = tune_thresholds(vec_val, map_result, fold_a)
        threshold_for_a = tune_thresholds(vec_val, map_result, fold_b)
        print(f"thresholds (tuned on even, applied to odd): {threshold_for_b}", file=sys.stderr)
        print(f"thresholds (tuned on odd, applied to even): {threshold_for_a}", file=sys.stderr)
        vec_song_eval = evaluate_fold(vec_val, map_result, fold_a, threshold_for_a, args.top2) \
            + evaluate_fold(vec_val, map_result, fold_b, threshold_for_b, args.top2)
        protocol["thresholds"] = {"for_even": threshold_for_a, "for_odd": threshold_for_b}

    report(vec_song_eval)
    if args.records:
        dump_records(args.records, protocol, vec_song_eval)
    if args.save_calibration:
        vec_threshold = tune_thresholds(vec_val, map_result, list(range(len(vec_val))))
        path_calibration = args.ckpt + ".calibration.json"
        with open(path_calibration, "w", encoding="utf-8") as handle:
            json.dump({"thresholds": vec_threshold, "top2_hands": True,
                       "tuned_on": f"val x{len(vec_val)}", "ckpt": args.ckpt}, handle, indent=1)
        print(f"calibration -> {path_calibration}  thresholds={vec_threshold}", file=sys.stderr)


if __name__ == "__main__":
    main()
