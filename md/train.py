"""Training entry for the f4 detection baseline: random-window training, per-epoch whole-song
tiled validation with event-F1 reporting (md.evaluate), last checkpoint saved.

Usage (runs on lab -- the feature cache lives there):
  python -m md.train --manic <manic.sqlite> --audio-db <manic_audio.sqlite> [--epochs 3]
                     [--batch 16] [--windows-per-song 4] [--limit-songs 0] [--out .out/md_f4_smoke.pt]
"""
import argparse
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dio.common_struct import N_LANE, PEDAL_NONE, PEDAL_UNKNOWN

from .dataset import TATUM_PER_BEAT, WindowDataset, load_song_entries, split_entries
from .detector import DrumDetector, phase_weight
from .evaluate import evaluate_song, report
from .infer import decode_onsets, infer_song_probs

SEED = 7


def onset_pos_weight(vec_entry) -> float:
    """BCE pos_weight from the corpus onset density, sqrt-tempered and capped (a raw neg/pos of ~35
    would trade all precision for recall; the phase weights already push the common positions)"""
    cnt_pos = sum(int(entry.arr_onset.sum()) for entry in vec_entry)
    cnt_all = sum(entry.arr_onset.size for entry in vec_entry)
    return float(min(10.0, max(1.0, ((cnt_all - cnt_pos) / max(1, cnt_pos)) ** 0.5)))


def loss_of(output: dict, batch: dict, pos_weight: torch.Tensor) -> torch.Tensor:
    arr_tatum = batch["tatum_lo"][:, None] + torch.arange(batch["onset"].shape[1], device=pos_weight.device)[None, :]
    weight = phase_weight(arr_tatum)[:, :, None]
    loss_onset = (nn.functional.binary_cross_entropy_with_logits(
        output["onset"], batch["onset"], pos_weight=pos_weight, reduction="none") * weight).mean()

    mask_onset = batch["onset"] > 0
    loss_velocity = nn.functional.mse_loss(output["velocity"][mask_onset], batch["velocity"][mask_onset]) \
        if mask_onset.any() else output["velocity"].sum() * 0.0

    mask_pedal = (batch["pedal"] != PEDAL_NONE) & (batch["pedal"] != PEDAL_UNKNOWN)
    loss_pedal = nn.functional.cross_entropy(output["pedal"][mask_pedal], batch["pedal"][mask_pedal]) \
        if mask_pedal.any() else output["pedal"].sum() * 0.0
    return loss_onset + 0.5 * loss_velocity + 0.2 * loss_pedal


def validate(model, vec_entry_val, path_audio_db, str_device, batch_size) -> dict:
    """Tiled whole-song inference -> decode at flat 0.5 (training-time quick look; the calibrated
    protocol lives in md.eval_ckpt) -> event-F1 report"""
    map_result = infer_song_probs(model, vec_entry_val, path_audio_db, str_device, batch_size)
    vec_song_eval = []
    for index, entry in enumerate(vec_entry_val):
        result = map_result[index]
        arr_onset = decode_onsets(result["onset_prob"], [0.5] * N_LANE)
        vec_song_eval.append(evaluate_song(entry.audio_key, entry.time_map, entry.arr_onset,
                                           entry.arr_velocity, entry.arr_pedal, arr_onset,
                                           result["velocity"], result["pedal"], TATUM_PER_BEAT))
    return report(vec_song_eval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manic", required=True)
    parser.add_argument("--audio-db", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--windows-per-song", type=int, default=4)
    parser.add_argument("--limit-songs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=".out/md_f4_smoke.pt")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    vec_entry = load_song_entries(args.manic, args.audio_db)
    if args.limit_songs > 0:
        vec_entry = vec_entry[:args.limit_songs]
    vec_train, vec_val = split_entries(vec_entry)
    print(f"songs: train={len(vec_train)} val={len(vec_val)}", file=sys.stderr)

    model = DrumDetector().to(args.device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", file=sys.stderr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pos_weight = torch.tensor(onset_pos_weight(vec_train), device=args.device)
    print(f"onset pos_weight: {float(pos_weight):.2f}", file=sys.stderr)

    loader = DataLoader(WindowDataset(vec_train, args.audio_db, windows_per_song=args.windows_per_song),
                        batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    for epoch in range(1, args.epochs + 1):
        time_begin = time.time()
        loss_sum, cnt_step = 0.0, 0
        for batch in loader:
            batch = {key: value.to(args.device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            loss = loss_of(model(batch["feat"], batch["tatum_lo"]), batch, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss)
            cnt_step += 1
        print(f"epoch {epoch}: loss {loss_sum / max(1, cnt_step):.4f} "
              f"({cnt_step} steps, {time.time() - time_begin:.0f}s)", file=sys.stderr)
        validate(model, vec_val, args.audio_db, args.device, args.batch)

    torch.save({"model": model.state_dict(), "epochs": args.epochs, "seed": SEED}, args.out)
    print(f"saved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
