"""Backups and reverting for Asset Swapper mods.

The tool never changes the game's own files: everything it makes lives in a mod folder under the game's `mods` folder. So the
game is back to the original when the mod is switched off or removed. This module keeps copies of what was there before every
change, so a build, a restore or a removal can be undone:

    <mods>/_asset-swap-backups/<mod>/<YYYYmmdd-HHMMSS>/   a copy of the mod folder as it was, plus backup.json

(The game's mod loader leaves out folders whose names start with `_` or `-`, so the backups are never loaded.)
"""
import json
import os
import shutil
import time

DIRNAME = '_asset-swap-backups'
KEEP = 12                      # snapshots kept per mod; older ones are pruned


def root(mods_dir):
    return os.path.join(mods_dir, DIRNAME)


def _files(path):
    n = 0
    for _d, _s, fs in os.walk(path):
        n += len(fs)
    return n


def _active_path(mods_dir, mod):
    """The mod's folder, whether it is on (`name`) or off (`-name`)."""
    for cand in (mod, '-' + mod):
        p = os.path.join(mods_dir, cand)
        if os.path.isdir(p):
            return p
    return None


def snapshot(mods_dir, mod, reason=''):
    """Copy the mod's folder into a new timestamped backup. Returns its path, or None when there is nothing to back up."""
    src = _active_path(mods_dir, mod)
    if not src or not _files(src):
        return None
    stamp = time.strftime('%Y%m%d-%H%M%S')
    dst = os.path.join(root(mods_dir), mod, stamp)
    n = 1
    while os.path.exists(dst):                                   # two within a second
        dst = os.path.join(root(mods_dir), mod, '%s-%d' % (stamp, n))
        n += 1
    shutil.copytree(src, dst)
    json.dump({'mod': mod, 'reason': reason, 'time': stamp, 'files': _files(src), 'was_on': os.path.basename(src) == mod},
              open(os.path.join(dst, 'backup.json'), 'w'), indent=1)
    prune(mods_dir, mod)
    return dst


def list_snapshots(mods_dir, mod=None):
    """[(mod, stamp, path, meta)] newest first."""
    out = []
    r = root(mods_dir)
    if not os.path.isdir(r):
        return out
    for m in sorted(os.listdir(r)):
        if mod and m != mod:
            continue
        md = os.path.join(r, m)
        for s in sorted(os.listdir(md), reverse=True):
            p = os.path.join(md, s)
            meta = {}
            try:
                meta = json.load(open(os.path.join(p, 'backup.json')))
            except (OSError, ValueError):
                pass
            out.append((m, s, p, meta))
    return sorted(out, key=lambda x: x[1], reverse=True)


def prune(mods_dir, mod, keep=KEEP):
    snaps = list_snapshots(mods_dir, mod)
    for _m, _s, p, _meta in snaps[keep:]:
        shutil.rmtree(p, ignore_errors=True)


def restore(mods_dir, snap_path):
    """Put a backup back as the mod. What is there now is backed up first, so a restore can itself be undone."""
    meta = json.load(open(os.path.join(snap_path, 'backup.json')))
    mod = meta['mod']
    snapshot(mods_dir, mod, 'before restoring %s' % os.path.basename(snap_path))
    cur = _active_path(mods_dir, mod)
    if cur:
        shutil.rmtree(cur)
    dest = os.path.join(mods_dir, mod if meta.get('was_on', True) else '-' + mod)
    shutil.copytree(snap_path, dest, ignore=shutil.ignore_patterns('backup.json'))
    return dest


def set_enabled(mods_dir, mod, on):
    """Switch a mod on or off by renaming its folder (a leading '-' switches it off). Returns the new path."""
    cur = _active_path(mods_dir, mod)
    if not cur:
        raise FileNotFoundError(mod)
    new = os.path.join(mods_dir, mod if on else '-' + mod)
    if cur != new:
        os.rename(cur, new)
    return new


def remove(mods_dir, mod):
    """Remove the mod from the game (the game is then as it was). A backup is kept first. Returns the backup path."""
    cur = _active_path(mods_dir, mod)
    if not cur:
        raise FileNotFoundError(mod)
    snap = snapshot(mods_dir, mod, 'before removing the mod')
    shutil.rmtree(cur)
    return snap


def delete_snapshot(snap_path):
    shutil.rmtree(snap_path, ignore_errors=True)


def mods(mods_dir):
    """[(name, on)] for the mod folders in the mods folder (backups excluded)."""
    out = []
    if not os.path.isdir(mods_dir):
        return out
    for n in sorted(os.listdir(mods_dir)):
        p = os.path.join(mods_dir, n)
        if not os.path.isdir(p) or n == DIRNAME or n.startswith('_'):
            continue
        out.append((n[1:] if n.startswith('-') else n, not n.startswith('-')))
    return out
