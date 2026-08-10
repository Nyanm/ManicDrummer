"""mdb_fz.xml parser -- the game-side per-song labels dio.build joins onto charts.

xg_diff_list slot layout validated by m1f0s6 (MAS slot <-> difficulty-4 chunk agreement 1355/1357):
u16[15] fixed-point (x100), grouped [drum, guitar, bass] x [slot0, BSC, ADV, EXT, MAS]. Only the
drum group is carried here. CLASSIC re-entries share a seq_id; the first record wins (GitadoraOnEar
convention: the lower music_id is the original)."""
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field


@dataclass
class MdbRecord:
    music_id: int
    seq_id: int
    title: str
    bpm: int
    bpm2: int
    vec_drum_difficulty: list = field(default_factory=list)  # 5 slots [slot0, BSC, ADV, EXT, MAS], x100 fixed-point


def _text_of(node, tag: str) -> str:
    child = node.find(tag)
    return child.text.strip() if child is not None and child.text else ""


def parse_mdb(path_mdb) -> dict:
    """Parse mdb_fz.xml into {seq_id: MdbRecord} (first record per seq_id wins)"""
    root = ElementTree.parse(str(path_mdb)).getroot()
    map_seq_record = {}
    for node in root.iter("mdb_data"):
        vec_diff = [int(token) for token in _text_of(node, "xg_diff_list").split()]
        if len(vec_diff) != 15:
            continue
        record = MdbRecord(
            music_id=int(_text_of(node, "music_id")),
            seq_id=int(_text_of(node, "seq_id")),
            title=_text_of(node, "title_name"),
            bpm=int(_text_of(node, "bpm") or 0),
            bpm2=int(_text_of(node, "bpm2") or 0),
            vec_drum_difficulty=vec_diff[0:5],
        )
        map_seq_record.setdefault(record.seq_id, record)
    return map_seq_record


def difnum_of(record: MdbRecord, difficulty: int) -> int:
    """The x100 fixed-point difficulty value for a chart's SQ3T difficulty byte (1=BSC..4=MAS);
    0 when the slot is empty or the byte is outside the labelled range (the difficulty-0 oddballs)."""
    return record.vec_drum_difficulty[difficulty] if 1 <= difficulty <= 4 else 0
