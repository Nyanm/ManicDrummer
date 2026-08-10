"""Instrument semantics derived from keysound names. Names span several eras of conventions
("036_BD1", "KICK1_C", "HH_PEDAL", "RIDE_D#", "SPU1027D...") -- 3643 raw classes library-wide
(m1f0s4), so nothing here claims exact taxonomy: lanes are the trusted signal, names only add the
semantics lanes cannot carry (leftpedal HH/BD split, exotic instrumentation, name usability)."""
import re

from .common_struct import PEDAL_BD, PEDAL_HH, PEDAL_UNKNOWN

_RE_NAME_PREFIX = re.compile(r"^\d+_")
_RE_UNNAMED = re.compile(r"^SPU\d+")
_RE_PITCH_SUFFIX = re.compile(r"_[A-G]#?$")
_RE_SIDE_SUFFIX = re.compile(r"_?(L|R)$")
_RE_TRAILING_JUNK = re.compile(r"[\d_.]+$")
_RE_KICK_FAMILY = re.compile(r"KICK|BD")
_RE_PEDAL_HH_FAMILY = re.compile(r"FHH|HHF|PHH|PEDAL|HH")

_EXOTIC_TOKENS = ("TAIKO", "CONGA", "BONGO", "TABLA", "TIMBAL", "COWBELL", "CLAP", "SHAKER", "TAMB",
                  "PERC", "MARIMBA", "DJEMBE", "CAJON", "AGOGO", "GONG", "WHISTLE", "VIBRASLAP",
                  "GUIRO", "CLAVE", "WOODBLOCK", "CASTANET", "TRIANGLE")

CLASS_UNNAMED = "UNNAMED"


def classify_name(name_keysound: str) -> str:
    """Normalise a raw keysound name to its class: numeric prefix, pitch suffixes (_C.._G#), side
    suffixes (_L/_R) and trailing digits/underscores are stripped in a loop until stable ("KICK1_C"
    -> "KICK"); filename-like names (SPU****D...) carry no identity and collapse to UNNAMED."""
    stem = name_keysound.upper()
    if _RE_UNNAMED.match(stem):
        return CLASS_UNNAMED
    stem = _RE_NAME_PREFIX.sub("", stem)
    while True:
        shorter = _RE_TRAILING_JUNK.sub("", _RE_PITCH_SUFFIX.sub("", stem))
        if shorter == stem:
            break
        stem = shorter
    stem = _RE_SIDE_SUFFIX.sub("", stem)
    return stem or CLASS_UNNAMED


def is_exotic(name_class: str) -> bool:
    """True when the class names non-kit percussion -- the variant-instrumentation signal (m1f0s4)"""
    return any(token in name_class for token in _EXOTIC_TOKENS)


def pedal_semantic(name_class: str, sound_id: int, set_sid_kick_lane: set) -> int:
    """Per-note semantic of a leftpedal (lane 8) note. Double criterion, kick first: the same
    sound_id being playable on the bass lane anywhere in the song is a stronger kick signal than any
    name; the HH regex is deliberately wide (FHH/HHF/PHH/PEDAL/HH) and only consulted after it."""
    if sound_id in set_sid_kick_lane or _RE_KICK_FAMILY.search(name_class):
        return PEDAL_BD
    if _RE_PEDAL_HH_FAMILY.search(name_class):
        return PEDAL_HH
    return PEDAL_UNKNOWN
