"""Locate each song's rendered Opus in the GitadoraOnEar music library by its TITLE tag.

The library is organised {version album dir}/{cleaned title}.opus; filenames went through Windows
character cleaning, so matching on the embedded TITLE tag (written verbatim from the same mdb this
project reads) is the lossless join. Paths returned are RELATIVE to the music root -- manic.sqlite
never stores an absolute path (see ENVIRONMENT, audio_path lesson)."""
from pathlib import Path

from mutagen.oggopus import OggOpus

_DIR_PREFIXES_GITADORA = ("GFDM ", "GITADORA ")


def build_title_index(path_music_root) -> tuple[dict, int]:
    """Scan the GITADORA album dirs under the music root and map TITLE tag -> relative path.
    Returns (map_title_relpath, cnt_collision); on a title collision the first hit wins."""
    path_music_root = Path(path_music_root)
    map_title_relpath, cnt_collision = {}, 0
    for dir_album in sorted(path_music_root.iterdir()):
        if not dir_album.is_dir() or not dir_album.name.startswith(_DIR_PREFIXES_GITADORA):
            continue
        for path_opus in sorted(dir_album.glob("*.opus")):
            vec_title = OggOpus(str(path_opus)).get("title")
            if not vec_title:
                continue
            title = vec_title[0]
            if title in map_title_relpath:
                cnt_collision += 1
                continue
            map_title_relpath[title] = path_opus.relative_to(path_music_root).as_posix()
    return map_title_relpath, cnt_collision
