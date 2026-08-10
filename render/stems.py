"""Part-combination rendering: drum stems from keysounds, accompaniment beds from the bgm archive.

The audio truth of the drum part does not depend on difficulty (difficulties re-select lanes/auto
over the same sounds), so a stem renders the UNION of every chart's (tick, sound_id) events plus the
metadata chunk's auto-play notes -- the same set the game engine sounds. Per-note gain is the
audited model from render.mix; `note_gain_fn` is the augmentation hook (returns a multiplier per
note, e.g. velocity jitter or kit-piece dropout for training-time remixing).

Beds (`m<id>_bgm.ifs` members) are named bgm<id><d|_><g|_><b|_>k[.bin|_xg.bin] -- the mask says
which parts are ALREADY baked in. A no-drum accompaniment is the bed with drums absent and the most
other parts present (usually `_gbk`); full mix = that bed + the rendered drum stem."""
from dataclasses import dataclass

import numpy as np
from ifstools import IFS

from dio.common_struct import DrumNote, TICK_PER_SECOND
from dio.sq3 import parse_drum_meta_notes, parse_drum_sq3
from dio.va3 import parse_data_start, parse_keysound_table, payload_of

from .adpcm import decode_mono
from .bmp import BmpAudio, decode_bmp
from .mix import KEYSOUND_GAIN, Timeline, VOLUME_FULL_SCALE, pan_gains

RATE_HZ_NATIVE = 48000


@dataclass
class DrumKit:
    """A decoded drum keysound bank: sound_id -> mono PCM plus its archive-side volume/pan"""
    map_pcm: dict      # {sound_id: (samples,) int16}
    map_entry: dict    # {sound_id: KeysoundEntry}


def load_drum_kit(bytes_va3: bytes) -> DrumKit:
    data_start = parse_data_start(bytes_va3)
    map_pcm, map_entry = {}, {}
    for entry in parse_keysound_table(bytes_va3):
        map_pcm[entry.sound_id] = decode_mono(payload_of(bytes_va3, data_start, entry))
        map_entry[entry.sound_id] = entry
    return DrumKit(map_pcm=map_pcm, map_entry=map_entry)


def drum_sound_events(bytes_sq3: bytes, include_auto: bool = True) -> list:
    """The audio-truth event set: union of every difficulty's notes deduplicated on (tick,
    sound_id), plus metadata auto-play notes. With include_auto=False only notes playable in at
    least one difficulty remain (the sounding contribution of the pads themselves)."""
    _, _, vec_chart = parse_drum_sq3(bytes_sq3)
    map_key_note = {}
    for chart in vec_chart:
        for note in chart.vec_note:
            existing = map_key_note.get((note.tick, note.sound_id))
            if existing is None or (existing.is_auto and not note.is_auto):
                map_key_note[(note.tick, note.sound_id)] = note  # playable wins over auto duplicates
    for note in parse_drum_meta_notes(bytes_sq3):
        map_key_note.setdefault((note.tick, note.sound_id), note)
    vec_event = [note for note in map_key_note.values() if include_auto or not note.is_auto]
    vec_event.sort(key=lambda note: note.tick)
    return vec_event


def render_drum_stem(bytes_sq3: bytes, bytes_va3: bytes, include_auto: bool = True,
                     keysound_gain: float = KEYSOUND_GAIN, note_gain_fn=None,
                     rate_hz: int = RATE_HZ_NATIVE) -> Timeline:
    """Render the drum layer alone onto a silent timeline (no bed, no limiter -- the caller stages
    the result into a mix or normalises it for standalone use)."""
    kit = load_drum_kit(bytes_va3)
    _, timing, _ = parse_drum_sq3(bytes_sq3)
    timeline = Timeline(rate_hz, frames_hint=int(timing.end_tick * rate_hz // TICK_PER_SECOND) + rate_hz)
    cnt_unresolved = 0
    for note in drum_sound_events(bytes_sq3, include_auto=include_auto):
        entry = kit.map_entry.get(note.sound_id)
        if entry is None:
            cnt_unresolved += 1  # empty sound ids (0/1/2) occur in 21 songs, expected (m4f7s2)
            continue
        frame_start = note.tick * rate_hz // TICK_PER_SECOND
        gain = keysound_gain * (entry.volume / VOLUME_FULL_SCALE) * (note.velocity / VOLUME_FULL_SCALE)
        if note_gain_fn is not None:
            gain *= note_gain_fn(note)
        gain_left, gain_right = pan_gains(entry.pan)
        timeline.add_mono_i16(frame_start, kit.map_pcm[note.sound_id], gain * gain_left, gain * gain_right)
    return timeline


def bed_mask_of(name_member: str):
    """Parse a bed member name into (has_drum, has_guitar, has_bass), or None for non-bed members
    (the 10-second i<id>dm.bin previews must not match)."""
    if not name_member.startswith("bgm"):
        return None
    stem = name_member.removesuffix("_xg.bin") if name_member.endswith("_xg.bin") \
        else name_member.removesuffix(".bin")
    if stem == name_member or len(stem) < 4 or stem[-1] != "k":
        return None
    mask = stem[-4:-1]
    for slot, letter in zip(mask, "dgb"):
        if slot not in ("_", letter):
            return None
    return mask[0] == "d", mask[1] == "g", mask[2] == "b"


def list_beds(path_bgm_ifs) -> dict:
    """The bed members of a bgm .ifs: {member_name: (has_drum, has_guitar, has_bass)}"""
    archive = IFS(str(path_bgm_ifs))
    map_name_mask = {}
    for name, _item in archive.tree.files.items():
        mask = bed_mask_of(name)
        if mask is not None:
            map_name_mask[name] = mask
    return map_name_mask


def load_accompaniment(path_bgm_ifs, want_drums: bool = False) -> tuple[str, BmpAudio]:
    """Load the bed best matching `want_drums` on the drum slot while carrying the MOST other parts
    (a no-drum accompaniment should still contain guitar and bass). Returns (member_name, audio)."""
    map_name_mask = list_beds(path_bgm_ifs)
    if not map_name_mask:
        raise FileNotFoundError(f"no bed member in {path_bgm_ifs}")
    name_best = max(map_name_mask,
                    key=lambda name: (map_name_mask[name][0] == want_drums,
                                      map_name_mask[name][1] + map_name_mask[name][2]))
    archive = IFS(str(path_bgm_ifs))
    return name_best, decode_bmp(archive.tree.files[name_best].load(convert=False))
