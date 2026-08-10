"""SEQP / SQ3T drum chart parser (`d<id>.sq3`) -> TimingMap + per-difficulty DrumCharts.

Field layout frozen by iidxOnKnitting m4f0#f0s2 and re-validated on this library (m1f0):
container SEQP (0x10 data_offset, 0x14 music_id, 0x18 chunk count), chunks of 0x10-byte header +
SQ3T body, events of 0x40 bytes. Unlike the audio-side parser (which unions difficulties), charts
are kept PER DIFFICULTY -- lane and auto assignments are what differ between them (m1f0s3)."""
import struct

from .common_struct import BpmChange, DrumChart, DrumNote, LANE_AUTO, TimingMap

_SEQP_MAGIC = b"SEQP"
_SQ3T_MAGIC = b"SQ3T"

_DATA_OFFSET_AT = 0x10
_MUSIC_ID_AT = 0x14
_CHART_COUNT_AT = 0x18
_CHUNK_HEADER_LEN = 0x10

# SQ3T chunk header, relative to the chunk body
_HEADER_SIZE_AT = 0x0C
_EVENT_COUNT_AT = 0x10
_IS_METADATA_AT = 0x15
_DIFFICULTY_AT = 0x16
_GAME_TYPE_AT = 0x17
_TIME_DIVISION_AT = 0x18
_BEAT_DIVISION_AT = 0x1A
_EVENT_SIZE_AT = 0x1C
_SQ3T_HEADER_END = 0x20

# event block offsets
_TICK_AT = 0x00
_EVENT_ID_AT = 0x04
_BEAT_AT = 0x10
_SOUND_ID_AT = 0x20
_VOLUME_AT = 0x2D
_LANE_AT = 0x30
_AUTO_AT = 0x34          # note events: auto flag byte; bpm events: u32 microseconds per beat

_EVENT_BPM = 0x01
_EVENT_MEASURE = 0x05
_EVENT_BEAT = 0x06
_EVENT_ENDPOS = 0x0F
_EVENT_NOTE = 0x10

_GAME_TYPE_DRUM = 0


class Sq3ParseError(ValueError):
    pass


def parse_drum_sq3(bytes_sq3: bytes) -> tuple[int, TimingMap, list]:
    """Parse a d<id>.sq3 into (music_id, TimingMap, [DrumChart]). Non-SQ3T chunks (older SEQT,
    padding) are skipped; non-drum SQ3T chunks never occur in d-files but are guarded anyway."""
    if len(bytes_sq3) < 0x20 or bytes_sq3[0:4] != _SEQP_MAGIC:
        raise Sq3ParseError(f"not a SEQP container (magic {bytes_sq3[0:4]!r})")
    data_offset, music_id, count_charts = struct.unpack("<III", bytes_sq3[_DATA_OFFSET_AT:_CHART_COUNT_AT + 4])

    timing = TimingMap()
    vec_chart = []
    cursor = data_offset
    for index_chunk in range(count_charts):
        if cursor + _CHUNK_HEADER_LEN > len(bytes_sq3):
            raise Sq3ParseError(f"chunk {index_chunk} header at {cursor} exceeds file size {len(bytes_sq3)}")
        size_chunk, = struct.unpack("<I", bytes_sq3[cursor:cursor + 4])
        if size_chunk == 0:
            raise Sq3ParseError(f"chunk {index_chunk} declares a zero size")
        # a chunk's declared size covers its own header, so the last one can reach past EOF by 0x10
        body = bytes_sq3[cursor + _CHUNK_HEADER_LEN:min(cursor + _CHUNK_HEADER_LEN + size_chunk, len(bytes_sq3))]
        cursor += size_chunk
        if len(body) < _SQ3T_HEADER_END or body[0:4] != _SQ3T_MAGIC:
            continue
        _read_chunk(body, index_chunk, timing, vec_chart)

    if not vec_chart:
        raise Sq3ParseError("no playable drum SQ3T chunk in the container")
    map_beat_change = {change.beat: change for change in timing.vec_bpm}  # last wins across chunks too
    timing.vec_bpm = sorted(map_beat_change.values(), key=lambda change: change.beat)
    return music_id, timing, vec_chart


def _read_chunk(body: bytes, index_chunk: int, timing: TimingMap, vec_chart: list):
    header_size, = struct.unpack("<I", body[_HEADER_SIZE_AT:_HEADER_SIZE_AT + 4])
    count_events, = struct.unpack("<I", body[_EVENT_COUNT_AT:_EVENT_COUNT_AT + 4])
    is_metadata, difficulty, game_type = body[_IS_METADATA_AT], body[_DIFFICULTY_AT], body[_GAME_TYPE_AT]
    time_division, beat_division = struct.unpack("<HH", body[_TIME_DIVISION_AT:_TIME_DIVISION_AT + 4])
    size_event, = struct.unpack("<I", body[_EVENT_SIZE_AT:_EVENT_SIZE_AT + 4])
    if size_event <= _AUTO_AT:
        raise Sq3ParseError(f"chunk {index_chunk} event size {size_event} too small")
    if header_size + count_events * size_event > len(body):
        raise Sq3ParseError(f"chunk {index_chunk} event table exceeds chunk size {len(body)}")

    if is_metadata:
        timing.time_division, timing.beat_division = time_division, beat_division
        _read_metadata_events(body, header_size, count_events, size_event, timing)
        return
    if game_type != _GAME_TYPE_DRUM:
        return

    chart = DrumChart(difficulty=difficulty)
    for index_event in range(count_events):
        base = header_size + index_event * size_event
        if body[base + _EVENT_ID_AT] != _EVENT_NOTE:
            if body[base + _EVENT_ID_AT] == _EVENT_ENDPOS:
                tick, = struct.unpack("<I", body[base:base + 4])
                beat, = struct.unpack("<I", body[base + _BEAT_AT:base + _BEAT_AT + 4])
                timing.end_tick = max(timing.end_tick, tick)
                timing.end_beat = max(timing.end_beat, beat)
            continue
        tick, = struct.unpack("<I", body[base:base + 4])
        beat, = struct.unpack("<I", body[base + _BEAT_AT:base + _BEAT_AT + 4])
        sound_id, = struct.unpack("<I", body[base + _SOUND_ID_AT:base + _SOUND_ID_AT + 4])
        if sound_id > 0xFFFF:
            raise Sq3ParseError(f"chunk {index_chunk} note {index_event} sound_id {sound_id} not a VA3 id")
        lane = body[base + _LANE_AT]
        is_auto = lane == LANE_AUTO or body[base + _AUTO_AT] != 0
        chart.vec_note.append(DrumNote(tick=tick, beat=beat, lane=lane, sound_id=sound_id,
                                       velocity=body[base + _VOLUME_AT], is_auto=is_auto))
    vec_chart.append(chart)


def _read_metadata_events(body: bytes, header_size: int, count_events: int, size_event: int, timing: TimingMap):
    map_beat_bpm = {}  # beat -> BpmChange; the LAST event at a beat wins (m2381 opens with two at beat 0)
    for index_event in range(count_events):
        base = header_size + index_event * size_event
        event_id = body[base + _EVENT_ID_AT]
        tick, = struct.unpack("<I", body[base:base + 4])
        beat, = struct.unpack("<I", body[base + _BEAT_AT:base + _BEAT_AT + 4])
        if event_id == _EVENT_BPM:
            micro_per_beat, = struct.unpack("<I", body[base + _AUTO_AT:base + _AUTO_AT + 4])
            if micro_per_beat > 0:
                map_beat_bpm[beat] = BpmChange(tick=tick, beat=beat, bpm=60000000 / micro_per_beat)
        elif event_id == _EVENT_MEASURE:
            timing.vec_measure.append((tick, beat))
        elif event_id == _EVENT_BEAT:
            timing.vec_beat.append((tick, beat))
        elif event_id == _EVENT_ENDPOS:
            timing.end_tick = max(timing.end_tick, tick)
            timing.end_beat = max(timing.end_beat, beat)
    timing.vec_bpm.extend(map_beat_bpm.values())
