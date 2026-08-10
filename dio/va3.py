"""VA3W drum keysound archive parser (`spu<id>d.va3`) -- entry table only (names / volume / pan /
sound_id). Audio payload decoding belongs to the render module (f2), not here.

Layout frozen by iidxOnKnitting m4f0#f0s2: 0x40-byte entries at entry_start, sound_id at +0x1A,
name at +0x20. The GDX header block carries the pad default sound_ids; their semantic order is only
loosely verified (m1800: GDXG extras read floortom/leftcymbal/leftpedal), so they are exposed as a
raw list and consumers must not over-trust the labels."""
import struct

from .common_struct import KeysoundEntry

_VA3_MAGIC = b"VA3W"
_ENTRY_COUNT_AT = 0x08
_GDX_START_AT = 0x10
_ENTRY_START_AT = 0x14
_ENTRY_LEN = 0x40


class Va3ParseError(ValueError):
    pass


def parse_keysound_table(bytes_va3: bytes) -> list:
    """Parse the entry table into [KeysoundEntry] (file order)"""
    if len(bytes_va3) < 0x20 or bytes_va3[0:4] != _VA3_MAGIC:
        raise Va3ParseError(f"not a VA3W archive (magic {bytes_va3[0:4]!r})")
    entry_count, = struct.unpack("<I", bytes_va3[_ENTRY_COUNT_AT:_ENTRY_COUNT_AT + 4])
    entry_start, = struct.unpack("<I", bytes_va3[_ENTRY_START_AT:_ENTRY_START_AT + 4])

    vec_entry = []
    for index_entry in range(entry_count):
        base = entry_start + index_entry * _ENTRY_LEN
        volume, pan, sound_id = struct.unpack("<BBH", bytes_va3[base + 0x18:base + 0x1C])
        name = bytes_va3[base + 0x20:base + 0x40].split(b"\0")[0].decode("ascii", "replace")
        vec_entry.append(KeysoundEntry(sound_id=sound_id, name=name, volume=volume, pan=pan))
    return vec_entry


def parse_gdx_defaults(bytes_va3: bytes) -> list:
    """The GDX header's default pad sound_ids as a raw list (6 for GDXH, 9 for GDXG). The nominal
    order is hihat/snare/bass/hightom/lowtom/rightcymbal (+ floortom/leftcymbal/leftpedal)."""
    gdx_start, = struct.unpack("<I", bytes_va3[_GDX_START_AT:_GDX_START_AT + 4])
    magic = bytes_va3[gdx_start:gdx_start + 4]
    count = 9 if magic == b"GDXG" else 6
    return list(struct.unpack(f"<{count}H", bytes_va3[gdx_start + 4:gdx_start + 4 + count * 2]))
