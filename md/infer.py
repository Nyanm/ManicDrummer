"""Whole-song tiled inference and onset decoding, shared by training-time validation (md.train)
and offline checkpoint evaluation (md.eval_ckpt). Inference returns PROBABILITIES per song;
decoding (thresholds + the simultaneity constraint) is a separate, cheap, replayable step --
threshold sweeps must never re-run the model."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from dio.common_struct import N_LANE, VEC_LANE_HAND

from .dataset import WindowDataset

MAX_SIMULTANEOUS_HANDS = 2


def infer_song_probs(model, vec_entry: list, path_audio_db, str_device: str, batch_size: int,
                     window_beats: int = 16) -> dict:
    """Tiled whole-song inference -> {index_entry: dict(onset_prob [n,9], velocity [n,9], pedal [n])}.
    window_beats must match the training context (carried in the checkpoint since f1s3)."""
    loader = DataLoader(WindowDataset(vec_entry, path_audio_db, tiled=True,
                                      use_mel=getattr(model, "use_mel", False), window_beats=window_beats),
                        batch_size=batch_size, num_workers=2)
    map_result = {index: {"onset_prob": np.zeros_like(entry.arr_onset, dtype=np.float32),
                          "velocity": np.zeros_like(entry.arr_velocity),
                          "pedal": np.zeros_like(entry.arr_pedal)}
                  for index, entry in enumerate(vec_entry)}
    flag_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(str_device) if "mel" in batch else None
            output = model(batch["feat"].to(str_device), batch["tatum_lo"].to(str_device), mel=mel)
            arr_prob = torch.sigmoid(output["onset"]).cpu().numpy()
            arr_velocity = output["velocity"].cpu().numpy()
            arr_pedal = output["pedal"].argmax(dim=-1).cpu().numpy()
            for index_in_batch in range(len(arr_prob)):
                index_entry = int(batch["index_entry"][index_in_batch])
                tatum_lo = int(batch["tatum_lo"][index_in_batch])
                result = map_result[index_entry]
                take = min(arr_prob.shape[1], len(result["onset_prob"]) - tatum_lo)
                if take <= 0:
                    continue
                result["onset_prob"][tatum_lo:tatum_lo + take] = arr_prob[index_in_batch, :take]
                result["velocity"][tatum_lo:tatum_lo + take] = arr_velocity[index_in_batch, :take]
                result["pedal"][tatum_lo:tatum_lo + take] = arr_pedal[index_in_batch, :take]
    if flag_training:
        model.train()
    return map_result


def decode_onsets(arr_prob: np.ndarray, vec_threshold, top2: bool = False) -> np.ndarray:
    """Probabilities [n_tatum, 9] -> onset multi-hot. Per-lane thresholds; with top2, tatums holding
    more than two HAND-lane positives keep only the two highest-probability hands -- the game's
    physical constraint is about hands, feet lanes (bass / leftpedal) are exempt (m2f0s2:
    kick + snare + crash chords are legal and common)."""
    arr_threshold = np.asarray(vec_threshold, dtype=np.float32).reshape(1, N_LANE)
    arr_onset = (arr_prob >= arr_threshold).astype(np.uint8)
    if top2:
        arr_hand = arr_onset[:, VEC_LANE_HAND]
        for index_tatum in np.flatnonzero(arr_hand.sum(axis=1) > MAX_SIMULTANEOUS_HANDS):
            vec_lane = np.array(VEC_LANE_HAND)[np.flatnonzero(arr_hand[index_tatum])]
            vec_keep = vec_lane[np.argsort(arr_prob[index_tatum, vec_lane])[-MAX_SIMULTANEOUS_HANDS:]]
            arr_onset[index_tatum, VEC_LANE_HAND] = 0
            arr_onset[index_tatum, vec_keep] = 1
    return arr_onset
