"""Where the game is, remembered between runs (a small JSON file under %APPDATA%)."""
import json
import os
import sys


def _path():
    base = os.environ.get('APPDATA') or os.path.expanduser('~')
    return os.path.join(base, 'AssetSwapper', 'settings.json')


def load():
    try:
        return json.load(open(_path(), encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save(d):
    p = _path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(d, open(p, 'w', encoding='utf-8'), indent=1)


def find_game():
    """The game folder (the one holding the recomp's program, a romfs folder and the mods folder): where this program sits, the
    last one used, or a folder beside it that has a romfs. Returns None when none is found."""
    here = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
    parent = os.path.dirname(here)
    beside = []
    for root in (parent, os.path.dirname(parent)):             # the folders next to the program, one and two levels up
        try:
            beside += [os.path.join(root, d) for d in sorted(os.listdir(root))]
        except OSError:
            pass
    cands = [here, parent, load().get('game')] + beside
    for c in cands:
        if c and os.path.isdir(os.path.join(c, 'romfs')):
            return c
    return None
