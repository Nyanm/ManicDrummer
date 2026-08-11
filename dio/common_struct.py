"""Shared data classes and defining constants of the dio pipeline. Every cross-module structure lives
here; changing a constant in this file changes the data semantics of the whole project.

Time has two native units, both carried by every SQ3 event:
  tick  1/300 second  (wall clock)
  beat  1/480 beat    (musical grid, authored by the game -- validated to match tick within 1 tick)
The model-facing grid is 1/48 beat (GRID_PER_BEAT); one grid step = 10 native beat units."""
from dataclasses import dataclass, field

TICK_PER_SECOND = 300     # SQ3 native wall-clock unit
BEAT_DIVISION = 480       # SQ3 native musical unit (1/480 beat)
GRID_PER_BEAT = 48        # model-facing grid; BEAT_DIVISION // GRID_PER_BEAT = 10 native units per step
LANE_AUTO = 0xFF

# lane byte (event offset 0x30) -> pad, frozen by m1f0s2 (gitadora-customs cross-checked on the full library)
LANE_NAMES = ["hihat", "snare", "bass", "hightom", "lowtom", "rightcymbal", "leftcymbal", "floortom", "leftpedal"]
N_LANE = 9
LANE_BASS = 2
LANE_LEFT_PEDAL = 8

"""
The simultaneity constraint is about HANDS: lanes 2 (bass) and 8 (leftpedal) are FEET, so
kick + snare + crash three-lane chords are legal and common (m2f0s2 ground truth: 8% of onset
tatums hold 3-4 lanes overall, while >2 HAND lanes is 0.72%). Any decode rule or legality metric
must count over VEC_LANE_HAND, never over all nine lanes.
"""
VEC_LANE_HAND = [0, 1, 3, 4, 5, 6, 7]

# per-note semantic of a leftpedal (lane 8) note; PEDAL_NONE for every other lane
PEDAL_NONE, PEDAL_HH, PEDAL_BD, PEDAL_UNKNOWN = 0, 1, 2, 3


@dataclass
class BpmChange:
    tick: int
    beat: int
    bpm: float


@dataclass
class TimingMap:
    """The metadata chunk's authoritative timing: bpm segments plus authored bar/beat lines.
    vec_bpm is deduplicated on beat (the LAST event at a beat wins -- m2381 opens with two events at
    beat 0 and the game honours the later one)."""
    time_division: int = TICK_PER_SECOND
    beat_division: int = BEAT_DIVISION
    end_tick: int = 0
    end_beat: int = 0
    vec_bpm: list = field(default_factory=list)       # [BpmChange], beat-ascending
    vec_measure: list = field(default_factory=list)   # [(tick, beat)] bar lines
    vec_beat: list = field(default_factory=list)      # [(tick, beat)] beat lines


@dataclass
class DrumNote:
    tick: int
    beat: int
    lane: int        # raw lane byte, LANE_AUTO for auto rows
    sound_id: int
    velocity: int    # note volume byte 0..127
    is_auto: bool    # lane == LANE_AUTO or the auto flag byte set


@dataclass
class DrumChart:
    difficulty: int                                   # SQ3T chunk difficulty byte (observed 0..4)
    vec_note: list = field(default_factory=list)      # [DrumNote] in file order


@dataclass
class KeysoundEntry:
    sound_id: int
    name: str
    volume: int      # archive-side volume 0..127 (multiplied with the note's own velocity at play time)
    pan: int         # 64 = centre
    # audio payload locators (render side); zero-filled when only the chart side is in play
    offset: int = 0      # relative to the archive's data_start
    filesize: int = 0    # exact payload length; v0 archives' LAST entry may overflow EOF by <=16 bytes
    rate_hz: int = 0
    channels: int = 0


@dataclass
class SongIr:
    """One song's drum-side rich IR: authoritative timing, the drum keysound table and every
    difficulty's chart. This is the unit dio.build serialises (per chart row, see to_chart_dict)."""
    music_id: int
    seq_id: int
    timing: TimingMap
    vec_keysound: list = field(default_factory=list)  # [KeysoundEntry]
    vec_chart: list = field(default_factory=list)     # [DrumChart]

    def to_chart_dict(self, chart: DrumChart) -> dict:
        """The `ir_rich` msgpack blob for one difficulty row: song-level timing/keysounds are
        denormalised into every row so a row is self-contained."""
        return {
            "music_id": self.music_id,
            "seq_id": self.seq_id,
            "difficulty": chart.difficulty,
            "timing": {
                "time_division": self.timing.time_division,
                "beat_division": self.timing.beat_division,
                "end_tick": self.timing.end_tick,
                "end_beat": self.timing.end_beat,
                "vec_bpm": [[change.tick, change.beat, change.bpm] for change in self.timing.vec_bpm],
                "vec_measure": self.timing.vec_measure,
                "vec_beat": self.timing.vec_beat,
            },
            "vec_keysound": [[entry.sound_id, entry.name, entry.volume, entry.pan] for entry in self.vec_keysound],
            "vec_note": [[note.tick, note.beat, note.lane, note.sound_id, note.velocity, int(note.is_auto)]
                         for note in chart.vec_note],
        }


def timing_from_dict(data: dict) -> TimingMap:
    """Rebuild a TimingMap from a decoded ir_rich blob's `timing` section (for consumers)"""
    return TimingMap(
        time_division=data["time_division"], beat_division=data["beat_division"],
        end_tick=data["end_tick"], end_beat=data["end_beat"],
        vec_bpm=[BpmChange(tick, beat, bpm) for tick, beat, bpm in data["vec_bpm"]],
        vec_measure=[tuple(pair) for pair in data["vec_measure"]],
        vec_beat=[tuple(pair) for pair in data["vec_beat"]],
    )
