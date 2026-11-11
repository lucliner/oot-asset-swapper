"""Where the game is, remembered between runs (a small JSON file under %APPDATA%)."""
import json
import os
import sys


def _path():
    base = os.environ.get('APPDATA') or os.path.expanduser('~')
    return os.path.join(base, 'SheikahAssetSwapper', 'settings.json')


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
    """The game folder (the one holding sheikah.exe, a romfs folder and the mods folder): where the program sits, the last one
    used, or the usual place. Returns None when none is found."""
    here = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
    cands = [here, os.path.dirname(here), load().get('game'), r'E:\ZeldaRecomp\sheikah']
    for c in cands:
        if c and os.path.isdir(os.path.join(c, 'romfs')):
            return c
    return None
