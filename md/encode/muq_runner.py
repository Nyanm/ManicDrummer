"""MuQ wrapper: lazy-load the model once (process-wide singleton), run a mono 24kHz waveform through
it in fp32, and return every hidden-state layer stacked frame-major as [n_frame, n_layer, dim] --
the cache's storage layout. MuQ pools mel to 25Hz, so n_frame ~= round(samples / 960).

fp32 throughout (the model card warns fp16 can NaN). Audio up to MUQ_MAX_SECONDS single-passes;
LONGER audio is CHUNK-ENCODED (VFT M6 step 0d): overlapping chunks, each contributing only its
centre frames (the margins have one-sided context and are dropped), tiled exactly on the 25Hz frame
grid -- so over-length songs and small-VRAM cards work instead of OOMing on the conformer's O(T^2)
attention. Ported verbatim from VFT vft/encode/muq_runner.py (audited by its 0d equivalence probe)."""
import numpy as np
import torch

from .. import config

_model = None  # process-wide MuQ singleton (weights are large)


def _get_model(str_device: str):
    """Load MuQ once and cache it; subsequent calls reuse the resident model"""
    global _model
    if _model is None:
        from muq import MuQ
        _model = MuQ.from_pretrained(config.MUQ_MODEL_NAME).to(str_device).eval()
    return _model


def encode(wav_mono_24k, str_device: str = "cuda", chunk_seconds: int | None = None):
    """Run MuQ over a mono 24kHz waveform -> [n_frame, n_layer, dim] float32 (all hidden states).

    Single pass up to MUQ_MAX_SECONDS; longer audio is chunked per MUQ_CHUNK_SECONDS /
    MUQ_CHUNK_OVERLAP_SECONDS. chunk_seconds forces the chunked path at a custom length."""
    duration_s = len(wav_mono_24k) / config.MUQ_SAMPLE_RATE_HZ
    if chunk_seconds is None:
        if duration_s <= config.MUQ_MAX_SECONDS:
            return _single_pass(wav_mono_24k, str_device)
        chunk_seconds = config.MUQ_CHUNK_SECONDS
    if duration_s <= chunk_seconds:
        return _single_pass(wav_mono_24k, str_device)
    return _encode_chunked(wav_mono_24k, str_device, chunk_seconds, config.MUQ_CHUNK_OVERLAP_SECONDS)


def _single_pass(wav_mono_24k, str_device: str):
    model = _get_model(str_device)
    wavs = torch.from_numpy(wav_mono_24k).unsqueeze(0).to(str_device)  # [1, samples], fp32
    with torch.no_grad():
        output = model(wavs, output_hidden_states=True)  # hidden_states: n_layer x [1, n_frame, dim]
    # stack all layers on-device (one host transfer) -> frame-major [n_frame, n_layer, dim]
    return torch.stack(output.hidden_states, dim=2).squeeze(0).float().cpu().numpy()


def _encode_chunked(wav_mono_24k, str_device: str, chunk_s: int, overlap_s: int):
    """Overlap-chunked encode that tiles the 25Hz frame grid exactly. Chunk i covers samples
    [i*hop, i*hop + chunk) with hop = chunk - overlap; it contributes frames [margin, hop_frames +
    margin) of its output (first chunk from 0, last chunk to its end), margin = overlap/2 -- so
    consecutive kept ranges are contiguous and every kept frame saw >= overlap/2 seconds of true
    context on each side. Chunk boundaries are whole seconds = whole 25Hz frames, so no resampling
    is involved; a chunk producing fewer frames than expected would silently misalign the tiling,
    hence the hard assert."""
    sr, frame_hz, hop = config.MUQ_SAMPLE_RATE_HZ, config.MUQ_FRAME_HZ, config.MUQ_HOP_SAMPLES
    chunk_samples = chunk_s * sr
    hop_samples = (chunk_s - overlap_s) * sr
    hop_frames = (chunk_s - overlap_s) * frame_hz
    margin_frames = overlap_s * frame_hz // 2
    pieces = []
    start = 0
    while True:
        if start + chunk_samples >= len(wav_mono_24k):
            """
            Final chunk: NOT the remainder [start, end) -- a short runt would give its frames far
            less context than every other kept frame. Instead run a FULL-LENGTH chunk ending at the
            audio end (frame-aligned start, `hop` = samples per 25Hz frame), and keep exactly the
            frames the tiling cursor still owes.
            """
            tail_start = max(0, (len(wav_mono_24k) - chunk_samples) // hop * hop)
            out = _single_pass(wav_mono_24k[tail_start:], str_device)
            lo = (start // hop + (margin_frames if start > 0 else 0)) - tail_start // hop
            pieces.append(out[max(0, lo):])
            break
        out = _single_pass(wav_mono_24k[start:start + chunk_samples], str_device)
        lo = 0 if start == 0 else margin_frames
        hi = hop_frames + margin_frames
        assert out.shape[0] >= hi, f"chunk at {start} produced {out.shape[0]} frames, expected >= {hi}"
        pieces.append(out[lo:hi])
        start += hop_samples
    return np.concatenate(pieces, axis=0)
