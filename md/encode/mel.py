"""Log-mel sidecar (f1s2): the second input channel of the N2N recipe -- SSL features carry
semantics, log-mel carries raw spectral transients at a higher frame rate. One `.mel16` file per
song beside the MuQ features ([n_frame_mel, N_MEL] float16 @ MEL_FRAME_HZ), index-free: shape is
derived from the file size, resume is file existence, constants are stamped into the audio db meta.

Build (CPU-only, safe to run beside a GPU training):
  python -m md.encode.mel --manic <sqlite> --music-root <dir> --audio-db <sqlite>
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from msql import connect_ro
from msql.audio_db import AudioDb, features_dir

from .. import config

MEL_FRAME_HZ = 100                                   # hop 240 samples at 24kHz
N_MEL = 128
MEL_FILE_SUFFIX = ".mel16"
_HOP_SAMPLES = config.MUQ_SAMPLE_RATE_HZ // MEL_FRAME_HZ
_BYTES_PER_FRAME = N_MEL * 2


def compute_log_mel(wav_mono_24k: np.ndarray) -> np.ndarray:
    """Mono 24kHz waveform -> [n_frame, N_MEL] float32 log-mel (natural log, floored at -80 dB-ish)"""
    import librosa
    mel = librosa.feature.melspectrogram(y=wav_mono_24k, sr=config.MUQ_SAMPLE_RATE_HZ,
                                         n_fft=1024, hop_length=_HOP_SAMPLES, n_mels=N_MEL)
    return np.log(np.maximum(mel, 1e-8)).T.astype(np.float32)


def mel_path(path_audio_db, str_audio_key: str) -> Path:
    return features_dir(path_audio_db) / (str_audio_key + MEL_FILE_SUFFIX)


def read_mel_window(path_mel: Path, frame_lo: int, frame_hi: int) -> np.ndarray:
    """pread frames [frame_lo, frame_hi) -> [n, N_MEL] float16 (clamped; shape from file size)"""
    n_frame = path_mel.stat().st_size // _BYTES_PER_FRAME
    index_lo = max(0, min(frame_lo, n_frame - 1))
    index_hi = max(index_lo + 1, min(n_frame, frame_hi))
    with open(path_mel, "rb", buffering=0) as handle:
        handle.seek(index_lo * _BYTES_PER_FRAME)
        raw = handle.read((index_hi - index_lo) * _BYTES_PER_FRAME)
    return np.frombuffer(raw, dtype=np.float16).reshape(-1, N_MEL)


def run_build(path_manic, path_music_root, path_audio_db) -> dict:
    from . import audio_io
    conn = connect_ro(path_manic)
    db = AudioDb.open_write(path_audio_db)  # only to stamp meta; mel files are index-free
    db.set_meta("mel_frame_hz", MEL_FRAME_HZ)
    db.set_meta("n_mel", N_MEL)
    db.commit()
    db.close()

    vec_song = conn.execute("SELECT DISTINCT seq_id, audio_path FROM charts WHERE audio_path != '' "
                            "ORDER BY seq_id").fetchall()
    cnt_done = cnt_skip = cnt_fail = 0
    for index_song, (seq_id, audio_path) in enumerate(vec_song, start=1):
        audio_key = f"m{seq_id:04d}"
        path_out = mel_path(path_audio_db, audio_key)
        if path_out.exists():
            cnt_skip += 1
            continue
        time_begin = time.perf_counter()
        try:
            wav = audio_io.load_mono(str(Path(path_music_root) / audio_path))
            arr_mel = compute_log_mel(wav).astype(np.float16)
            path_tmp = path_out.with_name(path_out.name + ".tmp")
            path_tmp.write_bytes(np.ascontiguousarray(arr_mel).tobytes())
            path_tmp.replace(path_out)
            cnt_done += 1
            if index_song % 100 == 0:
                print(f"[{index_song}/{len(vec_song)}] {audio_key} n_frame={len(arr_mel)} "
                      f"{time.perf_counter() - time_begin:.1f}s", file=sys.stderr)
        except Exception as exc:
            cnt_fail += 1
            print(f"[{index_song}/{len(vec_song)}] {audio_key} FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"mel build: done={cnt_done} skipped={cnt_skip} failed={cnt_fail}", file=sys.stderr)
    return {"done": cnt_done, "skipped": cnt_skip, "failed": cnt_fail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manic", required=True)
    parser.add_argument("--music-root", required=True)
    parser.add_argument("--audio-db", required=True)
    args = parser.parse_args()
    stats = run_build(args.manic, args.music_root, args.audio_db)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
