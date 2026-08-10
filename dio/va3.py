"""VA3W drum keysound archive parser (`spu<id>d.va3`) -- entry table and payload slicing. Audio
DECODING (ADPCM -> PCM) belongs to the render package; byte-level format knowledge stays here.

Layout frozen by iidxOnKnitting m4f0#f0s2: 0x40-byte entries at entry_start, payload locators at
+0x00/+0x04, sound_id at +0x1A, name at +0x20. The GDX header block carries the pad default
sound_ids; their semantic order is only loosely verified (m1800: GDXG extras read
floortom/leftcymbal/leftpedal), so they are exposed as a raw list.

v0 archives' LAST entry can declare a filesize that overflows the file by at most one 16-byte
alignment unit (iidxOnKnitting m4f2: 365/1214165 entries, all v0, all last) -- payload_of truncates
to EOF and anything larger is a parse error."""
import struct

from .common_struct import KeysoundEntry

_VA3_MAGIC = b"VA3W"
_ENTRY_COUNT_AT = 0x08
_GDX_START_AT = 0x10
_ENTRY_START_AT = 0x14
_DATA_START_AT = 0x18
_ENTRY_LEN = 0x40
_OVERFLOW_LIMIT = 16


class Va3ParseError(ValueError):
    pass


def parse_keysound_table(bytes_va3: bytes) -> list:
    """Parse the entry table into [KeysoundEntry] (file order), payload locators included"""
    if len(bytes_va3) < 0x20 or bytes_va3[0:4] != _VA3_MAGIC:
        raise Va3ParseError(f"not a VA3W archive (magic {bytes_va3[0:4]!r})")
    entry_count, = struct.unpack("<I", bytes_va3[_ENTRY_COUNT_AT:_ENTRY_COUNT_AT + 4])
    entry_start, = struct.unpack("<I", bytes_va3[_ENTRY_START_AT:_ENTRY_START_AT + 4])

    vec_entry = []
    for index_entry in range(entry_count):
        base = entry_start + index_entry * _ENTRY_LEN
        offset, filesize, channels, bits, rate_hz = struct.unpack("<IIHHI", bytes_va3[base:base + 0x10])
        volume, pan, sound_id = struct.unpack("<BBH", bytes_va3[base + 0x18:base + 0x1C])
        name = bytes_va3[base + 0x20:base + 0x40].split(b"\0")[0].decode("ascii", "replace")
        vec_entry.append(KeysoundEntry(sound_id=sound_id, name=name, volume=volume, pan=pan,
                                       offset=offset, filesize=filesize, rate_hz=rate_hz, channels=channels))
    return vec_entry


def parse_data_start(bytes_va3: bytes) -> int:
    """Where the audio payloads begin; entry offsets are relative to this"""
    return struct.unpack("<I", bytes_va3[_DATA_START_AT:_DATA_START_AT + 4])[0]


def payload_of(bytes_va3: bytes, data_start: int, entry: KeysoundEntry) -> bytes:
    """One entry's raw ADPCM payload, with the v0 last-entry EOF overflow truncated"""
    begin = data_start + entry.offset
    end = begin + entry.filesize
    if end > len(bytes_va3):
        if end - len(bytes_va3) > _OVERFLOW_LIMIT:
            raise Va3ParseError(f"entry sound_id={entry.sound_id} payload [{begin}..{end}] "
                                f"overflows file size {len(bytes_va3)} by more than {_OVERFLOW_LIMIT}")
        end = len(bytes_va3)
    return bytes_va3[begin:end]


def parse_gdx_defaults(bytes_va3: bytes) -> list:
    """The GDX header's default pad sound_ids as a raw list (6 for GDXH, 9 for GDXG). The nominal
    order is hihat/snare/bass/hightom/lowtom/rightcymbal (+ floortom/leftcymbal/leftpedal)."""
    gdx_start, = struct.unpack("<I", bytes_va3[_GDX_START_AT:_GDX_START_AT + 4])
    magic = bytes_va3[gdx_start:gdx_start + 4]
    count = 9 if magic == b"GDXG" else 6
    return list(struct.unpack(f"<{count}H", bytes_va3[gdx_start + 4:gdx_start + 4 + count * 2]))
