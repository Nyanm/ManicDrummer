"""Training entry for the f4 detection baseline: random-window training, per-epoch whole-song
tiled validation with event-F1 reporting (md.evaluate), last checkpoint saved.

Usage (runs on lab -- the feature cache lives there):
  python -m md.train --manic <manic.sqlite> --audio-db <manic_audio.sqlite> [--epochs 3]
                     [--batch 16] [--windows-per-song 4] [--limit-songs 0] [--out .out/md_f4_smoke.pt]
"""
import argparse
import math
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


VEC_LANE_TOM = [3, 4, 7]
TOM_PHASE_FLOOR = 1.5    # f0s3: 17-18% of tom events sit on down-weighted fill positions
AUX_ANY_TOM = 0.3        # f0s3: family confusion dominates -- reward detecting "a tom" as such


def onset_pos_weight(vec_entry, per_lane: bool = False) -> torch.Tensor:
    """BCE pos_weight from the corpus onset density, sqrt-tempered and capped (a raw neg/pos of ~35
    would trade all precision for recall; the phase weights already push the common positions).
    per_lane (f1s4) computes one weight per lane so scarce lanes (toms) stop being drowned."""
    arr_pos = np.sum([entry.arr_onset.sum(axis=0) for entry in vec_entry], axis=0).astype(np.float64)
    cnt_all = sum(len(entry.arr_onset) for entry in vec_entry)
    arr_weight = np.clip(np.sqrt((cnt_all - arr_pos) / np.maximum(1.0, arr_pos)), 1.0, 10.0)
    if not per_lane:
        cnt_pos = arr_pos.sum()
        arr_weight = np.full(9, np.clip(((cnt_all * 9 - cnt_pos) / max(1.0, cnt_pos)) ** 0.5, 1.0, 10.0))
    return torch.from_numpy(arr_weight.astype(np.float32))


def loss_of(output: dict, batch: dict, pos_weight: torch.Tensor, tom_remedy: bool = False) -> torch.Tensor:
    arr_tatum = batch["tatum_lo"][:, None] + torch.arange(batch["onset"].shape[1], device=pos_weight.device)[None, :]
    weight = phase_weight(arr_tatum)[:, :, None]
    if tom_remedy:  # fills live on fine positions; do not let the phase prior suppress the weak family
        weight = weight.expand(-1, -1, batch["onset"].shape[2]).clone()
        weight[:, :, VEC_LANE_TOM] = weight[:, :, VEC_LANE_TOM].clamp(min=TOM_PHASE_FLOOR)
    loss_onset = (nn.functional.binary_cross_entropy_with_logits(
        output["onset"], batch["onset"], pos_weight=pos_weight, reduction="none") * weight).mean()
    if tom_remedy:  # family-level auxiliary: detect "a tom" even when the member is uncertain
        logit_any_tom = torch.logsumexp(output["onset"][:, :, VEC_LANE_TOM], dim=-1)
        target_any_tom = batch["onset"][:, :, VEC_LANE_TOM].amax(dim=-1)
        loss_onset = loss_onset + AUX_ANY_TOM * nn.functional.binary_cross_entropy_with_logits(
            logit_any_tom, target_any_tom)

    mask_onset = batch["onset"] > 0
    loss_velocity = nn.functional.mse_loss(output["velocity"][mask_onset], batch["velocity"][mask_onset]) \
        if mask_onset.any() else output["velocity"].sum() * 0.0

    mask_pedal = (batch["pedal"] != PEDAL_NONE) & (batch["pedal"] != PEDAL_UNKNOWN)
    loss_pedal = nn.functional.cross_entropy(output["pedal"][mask_pedal], batch["pedal"][mask_pedal]) \
        if mask_pedal.any() else output["pedal"].sum() * 0.0
    return loss_onset + 0.5 * loss_velocity + 0.2 * loss_pedal


def validate(model, vec_entry_val, path_audio_db, str_device, batch_size, window_beats: int = 16) -> dict:
    """Tiled whole-song inference -> decode at flat 0.5 (training-time quick look; the calibrated
    protocol lives in md.eval_ckpt) -> event-F1 report"""
    map_result = infer_song_probs(model, vec_entry_val, path_audio_db, str_device, batch_size, window_beats)
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
    parser.add_argument("--mel", action="store_true", help="add the log-mel sidecar branch (f1s2)")
    parser.add_argument("--window-beats", type=int, default=16, help="training context length (f1s3)")
    parser.add_argument("--tom-remedy", action="store_true",
                        help="per-lane pos_weight + tom phase floor + any-tom aux loss (f1s4)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=".out/md_run", help="checkpoint prefix; writes {out}_best.pt / {out}_last.pt")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    vec_entry = load_song_entries(args.manic, args.audio_db)
    if args.limit_songs > 0:
        vec_entry = vec_entry[:args.limit_songs]
    vec_train, vec_val = split_entries(vec_entry)
    print(f"songs: train={len(vec_train)} val={len(vec_val)}", file=sys.stderr)

    model = DrumDetector(use_mel=args.mel).to(args.device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M mel={args.mel} "
          f"window={args.window_beats} tom_remedy={args.tom_remedy}", file=sys.stderr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pos_weight = onset_pos_weight(vec_train, per_lane=args.tom_remedy).to(args.device)
    print(f"onset pos_weight: {[round(float(w), 2) for w in pos_weight]}", file=sys.stderr)

    loader = DataLoader(WindowDataset(vec_train, args.audio_db, windows_per_song=args.windows_per_song,
                                      use_mel=args.mel, window_beats=args.window_beats),
                        batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    steps_total = len(loader) * args.epochs
    steps_warmup = max(1, int(steps_total * 0.03))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / steps_warmup if step < steps_warmup
        else 0.5 * (1.0 + math.cos(math.pi * (step - steps_warmup) / max(1, steps_total - steps_warmup))))

    f1_best = -1.0
    for epoch in range(1, args.epochs + 1):
        time_begin = time.time()
        loss_sum, cnt_step = 0.0, 0
        for batch in loader:
            batch = {key: value.to(args.device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            loss = loss_of(model(batch["feat"], batch["tatum_lo"], mel=batch.get("mel")), batch, pos_weight,
                           tom_remedy=args.tom_remedy)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss)
            cnt_step += 1
        print(f"epoch {epoch}: loss {loss_sum / max(1, cnt_step):.4f} lr {scheduler.get_last_lr()[0]:.2e} "
              f"({cnt_step} steps, {time.time() - time_begin:.0f}s)", file=sys.stderr)
        summary = validate(model, vec_val, args.audio_db, args.device, args.batch, args.window_beats)
        state = {"model": model.state_dict(), "epoch": epoch, "seed": SEED,
                 "val_micro_f1": summary["micro_f1"], "use_mel": args.mel,
                 "window_beats": args.window_beats, "tom_remedy": args.tom_remedy}
        torch.save(state, f"{args.out}_last.pt")
        if summary["micro_f1"] > f1_best:  # quick-look protocol (flat 0.5); v2 calibration comes offline
            f1_best = summary["micro_f1"]
            torch.save(state, f"{args.out}_best.pt")
            print(f"  new best ({f1_best:.4f}) -> {args.out}_best.pt", file=sys.stderr)
    print(f"done; best val micro F1 {f1_best:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
