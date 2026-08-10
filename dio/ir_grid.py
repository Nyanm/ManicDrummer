"""ir_rich -> ir_grid: project one DrumChart onto the model-facing 1/48-beat grid.

The native beat field (1/480 beat) IS the authored musical grid, so quantisation is a division, not
a tempo integration: one grid step = 10 native units. Notes whose beat is not a multiple of 10 are
off the 1/48 grid (finer than 48ths or quintuplet-family); they are snapped to the nearest step and
flagged, and the residue kept, so nothing is silently lost. The grid row also carries the per-note
leftpedal semantic -- the one label lanes cannot express (m1f0s4: both-semantics songs are the
majority, 483/1358)."""
from .common_struct import BEAT_DIVISION, DrumChart, GRID_PER_BEAT, LANE_LEFT_PEDAL, PEDAL_NONE
from .instrument import classify_name, pedal_semantic

_NATIVE_PER_STEP = BEAT_DIVISION // GRID_PER_BEAT  # 10

# ir_grid vec_note row indices (columnar-ish compact rows, kept in one place for every consumer)
ROW_GRID, ROW_RESIDUE, ROW_LANE, ROW_VELOCITY, ROW_PEDAL, ROW_PLAYABLE = range(6)


def chart_to_grid(chart: DrumChart, map_sid_name: dict, set_sid_kick_lane: set) -> dict:
    """Project one chart to the `ir_grid` blob dict:
    vec_note rows [grid48, residue, lane, velocity, pedal, playable], grid-ascending stable order.
    residue = beat - grid48 * 10, in native 1/480 units, range [-4, 5] after nearest-step snapping."""
    vec_row = []
    cnt_offgrid = 0
    for note in chart.vec_note:
        grid48 = (note.beat + _NATIVE_PER_STEP // 2) // _NATIVE_PER_STEP
        residue = note.beat - grid48 * _NATIVE_PER_STEP
        cnt_offgrid += residue != 0
        pedal = PEDAL_NONE
        if not note.is_auto and note.lane == LANE_LEFT_PEDAL:
            pedal = pedal_semantic(classify_name(map_sid_name.get(note.sound_id, "")), note.sound_id,
                                   set_sid_kick_lane)
        vec_row.append([grid48, residue, note.lane, note.velocity, pedal, int(not note.is_auto)])
    vec_row.sort(key=lambda row: (row[ROW_GRID], row[ROW_LANE]))
    return {
        "grid_per_beat": GRID_PER_BEAT,
        "n_offgrid": cnt_offgrid,
        "vec_note": vec_row,
    }
