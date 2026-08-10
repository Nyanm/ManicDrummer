"""Konami's 4-bit ADPCM codec (decode only), ported from iidxOnKnitting src/audio/adpcm.rs
(itself validated bit-exact against fisyher/gitadora-customs' adpcmwavetool.cpp).

Pure nibble stream: no block headers, no predictor table; every stream starts at step_index = 0,
pcm = 0. Two packings that differ in more than channel count:
  stereo  one byte = ONE frame, high nibble left / low nibble right, INDEPENDENT channel state
  mono    one byte = TWO consecutive samples (high nibble first), one shared state

The decode loop is sequential by nature (state recurrence), so it runs through precomputed
(state, code) -> (delta, next_state) tables; ~3s per full-length stereo bed in CPython, negligible
for keysounds. Library-scale bed decoding is not an M1 workload; revisit speed only if it becomes one."""
import numpy as np

_STEPS = [
    256, 272, 304, 336, 368, 400, 448, 496, 544, 592, 656, 720,
    800, 880, 960, 1056, 1168, 1280, 1408, 1552, 1712, 1888, 2080, 2288,
    2512, 2768, 3040, 3344, 3680, 4048, 4464, 4912, 5392, 5936, 6528, 7184,
    7904, 8704, 9568, 10528, 11584, 12736, 14016, 15408, 16960, 18656, 20512, 22576,
    24832,
]
_CHANGES = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]
_INDEX_MAX = len(_STEPS) - 1

# (state 0..48, code 0..15) -> signed delta / next state, so the hot loop is two table reads
_TAB_DELTA = [[0] * 16 for _ in range(len(_STEPS))]
_TAB_NEXT = [[0] * 16 for _ in range(len(_STEPS))]
for _state, _step in enumerate(_STEPS):
    for _code in range(16):
        _delta = (_step >> 3) + (_step >> 2 if _code & 1 else 0) + (_step >> 1 if _code & 2 else 0) \
            + (_step if _code & 4 else 0)
        _TAB_DELTA[_state][_code] = -_delta if _code & 8 else _delta
        _TAB_NEXT[_state][_code] = min(max(_state + _CHANGES[_code], 0), _INDEX_MAX)


def _decode_nibbles(vec_nibble, out):
    """Run one decoder state over a nibble sequence, writing int16 into `out` (same length)"""
    tab_delta, tab_next = _TAB_DELTA, _TAB_NEXT
    state, pcm = 0, 0
    for index, code in enumerate(vec_nibble):
        pcm += tab_delta[state][code]
        if pcm > 32767:
            pcm = 32767
        elif pcm < -32768:
            pcm = -32768
        state = tab_next[state][code]
        out[index] = pcm


def decode_stereo(payload: bytes) -> np.ndarray:
    """Stereo payload -> (frames, 2) int16; one byte per frame, high nibble left"""
    codes = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty((len(codes), 2), dtype=np.int16)
    _decode_nibbles((codes >> 4).tolist(), out[:, 0])
    _decode_nibbles((codes & 0x0F).tolist(), out[:, 1])
    return out


def decode_mono(payload: bytes) -> np.ndarray:
    """Mono payload -> (samples,) int16; one byte per TWO samples, high nibble first, shared state"""
    codes = np.frombuffer(payload, dtype=np.uint8)
    interleaved = np.empty(len(codes) * 2, dtype=np.uint8)
    interleaved[0::2] = codes >> 4
    interleaved[1::2] = codes & 0x0F
    out = np.empty(len(interleaved), dtype=np.int16)
    _decode_nibbles(interleaved.tolist(), out)
    return out
