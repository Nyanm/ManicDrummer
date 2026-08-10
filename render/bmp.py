"""BMP audio container (`bgm*.bin` beds, `i*.bin` previews) -- mixed-endian header + Konami ADPCM
payload. Layout frozen by iidxOnKnitting m4f0#f0s2: magic "BMP\\0", data_size/loop/rate big-endian,
channels/bits little-endian, payload from 0x20."""
import struct
from dataclasses import dataclass

import numpy as np

from .adpcm import decode_mono, decode_stereo

_BMP_MAGIC = b"BMP\0"
_HEADER_LEN = 0x20


class BmpParseError(ValueError):
    pass


@dataclass
class BmpAudio:
    rate_hz: int
    samples: np.ndarray  # (frames, 2) int16 -- mono sources are upmixed to both channels


def decode_bmp(bytes_bmp: bytes) -> BmpAudio:
    if len(bytes_bmp) < _HEADER_LEN or bytes_bmp[0:4] != _BMP_MAGIC:
        raise BmpParseError(f"not a BMP audio file (magic {bytes_bmp[0:4]!r})")
    channels, bits = struct.unpack("<HH", bytes_bmp[0x10:0x14])
    rate_hz, = struct.unpack(">I", bytes_bmp[0x14:0x18])
    if bits != 16 or channels not in (1, 2):
        raise BmpParseError(f"unsupported BMP shape: channels={channels} bits={bits}")
    payload = bytes_bmp[_HEADER_LEN:]
    if channels == 2:
        samples = decode_stereo(payload)
    else:
        mono = decode_mono(payload)
        samples = np.repeat(mono[:, None], 2, axis=1)
    return BmpAudio(rate_hz=rate_hz, samples=samples)
