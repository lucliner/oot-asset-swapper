"""Asset Swapper: replace character models with models from elsewhere.

    python -I tools/assetswap inspect  <romfs> <Model/File.bfres.zs> [--bones]
    python -I tools/assetswap preview <romfs> <Model/File.bfres.zs> <out.png> [--yaw deg]
    python -I tools/assetswap swap <romfs> <Model/File.bfres.zs> <model.obj> --mod <mod folder> [--scale s] [--rotate-y deg] [--preview out.png]
    python -I tools/assetswap ui
    python -I tools/assetswap backup list|create|restore|on|off|remove --mods <mods folder> [--mod name] [--stamp when]
    python -I tools/assetswap export-obj <romfs> <Model/File.bfres.zs> <out.obj>

Run from the repository root. Reads your own romfs; writes only where you tell it.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # (python -I does not add the script's folder)

import numpy as np

from containers import Romfs   # noqa: E402
from bfres import Bfres        # noqa: E402
from skeleton import world_matrices  # noqa: E402


def load(romfs, rel):
    return Bfres(Romfs(romfs).read(rel))


def shape_positions(m, s, W):
    """Bind-pose model-space positions of a shape. Smooth-skinned vertices are stored in model space; a shape with no
    skinning (0) sits in its bone's space, and one with a single influence (1) in the space of the bone each vertex names (its `_i0` is a rigid matrix index)."""
    p = m.read_attr(s, '_p0')[:, :3].astype(np.float64)
    if s.skin_count > 1:
        return p
    h = np.c_[p, np.ones(len(p))]
    if s.skin_count == 0 or not s.skin_bones:
        return (h @ W[s.bone].T)[:, :3]
    vb = m.read_attr(s, '_i0')[:, 0].astype(int)
    out = np.empty_like(p)
    for k in np.unique(vb):
        sel = vb == k
        out[sel] = (h[sel] @ W[m.matrix_to_bone[k]].T)[:, :3]   # (the value is a rigid matrix index)
    return out


def cmd_inspect(a):
    m = load(a.romfs, a.model)
    print('%s: model %s, %d bones (%d smooth, %d rigid matrices), %d shapes, %d materials'
          % (m.name, m.model_name, len(m.bones), m.smooth_count, m.rigid_count, len(m.shapes), len(m.materials)))
    W = world_matrices(m)
    worst, odd = 0.0, []
    for mi, bi in enumerate(m.matrix_to_bone):
        inv = np.eye(4)
        inv[:3, :] = m.inv_model[mi]
        if mi >= m.smooth_count:
            continue                          # (rigid matrices carry no usable inverse; only smooth ones are checked)
        e = float(np.abs(W[bi] @ inv - np.eye(4)).max())
        if e > 1e-3:
            odd.append('%s(%s matrix %d)' % (m.bones[bi].name, 'smooth' if mi < m.smooth_count else 'rigid', mi))
        else:
            worst = max(worst, e)
    print('skeleton check: world * inverse bind = I for all but %d of the %d smooth matrices (worst %.1e)%s'
          % (len(odd), m.smooth_count, worst, ('; differ: ' + ', '.join(odd)) if odd else ''))
    if a.bones:
        for b in m.bones:
            print('  %3d %-28s parent %3d smooth %3d' % (b.index, b.name, b.parent, b.smooth))
    tot_v = tot_t = 0
    for s in m.shapes:
        fm = ' '.join('%s:%04x' % (x.name, x.fmt) for x in s.attrs)
        print('  shape %2d %-40s mat %-28s skin %d bones %2d  v %6d  tri %6d  [%s]'
              % (s.index, s.name[:40], m.materials[s.material][:28], s.skin_count, s.skin_bone_count,
                 s.vertex_count, s.index_count // 3, fm))
        tot_v += s.vertex_count
        tot_t += s.index_count // 3
    print('total %d vertices, %d triangles' % (tot_v, tot_t))


def cmd_export_obj(a):
    m = load(a.romfs, a.model)
    W = world_matrices(m)
    lines = []
    base = 1
    for s in m.shapes:
        if not s.vertex_count:
            continue
        p = shape_positions(m, s, W)
        uv = m.read_attr(s, '_u0')
        idx = m.indices(s).reshape(-1, 3)
        lines.append('o %s' % s.name)
        lines += ['v %.6f %.6f %.6f' % tuple(v) for v in p]
        if uv is not None:
            lines += ['vt %.6f %.6f' % (u, 1 - v) for u, v in uv[:, :2]]
        for t in idx + base:
            lines.append('f ' + ' '.join(('%d/%d' % (i, i)) if uv is not None else str(i) for i in t))
        base += len(p)
    open(a.out, 'w').write('\n'.join(lines) + '\n')
    print('wrote', a.out, '(%d shapes)' % len(m.shapes))


def model_triangles(m, pose=False):
    """All triangles of a model in its bind pose (or the test pose), model space: (n, 3, 3) float array."""
    W = world_matrices(m)
    if pose:
        from posing import pose_matrices, posed_positions
        M = pose_matrices(m, W)
    tris = []
    for s in m.shapes:
        if not s.vertex_count:
            continue
        p = shape_positions(m, s, W)
        if pose and s.skin_count > 1:
            p = posed_positions(m, s, p, M)
        tris.append(p[m.indices(s).reshape(-1, 3)])
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def render(t, out, yaw=0.0, size=900, pitch=0.0):
    """Flat-shaded software render of triangles (n, 3, 3), +y up, camera looking down -z. Returns a one-line summary."""
    from PIL import Image, ImageDraw
    y, p = np.radians(yaw), np.radians(pitch)
    R = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rp = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    t = t @ (Rp @ R).T
    lo, hi = t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)
    scale = size / max(hi[1] - lo[1], hi[0] - lo[0], 1e-6) * 0.9
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    shade = 0.35 + 0.65 * np.abs(n[:, 2])
    order = np.argsort(t[:, :, 2].mean(1))        # far to near
    W_, H_ = size, int(size * (hi[1] - lo[1]) / max(hi[1] - lo[1], hi[0] - lo[0]) * 0.9 / 0.9 + size * 0.1)
    img = Image.new('RGB', (W_, H_), (24, 24, 28))
    d = ImageDraw.Draw(img)
    cx = (lo[0] + hi[0]) / 2
    for i in order:
        pts = [((v[0] - cx) * scale + W_ / 2, H_ - (v[1] - lo[1]) * scale - size * 0.05) for v in t[i]]
        c = int(60 + 190 * shade[i])
        d.polygon(pts, fill=(c, c, int(c * 0.9)))
    img.save(out)
    return 'wrote %s (%d triangles; extents %s .. %s)' % (out, len(t), np.round(lo, 2), np.round(hi, 2))


def cmd_preview(a):
    print(render(model_triangles(load(a.romfs, a.model), a.pose), a.out, a.yaw, a.size, a.pitch))


def cmd_preview_import(a):
    from importers import load_obj
    meshes = load_obj(a.file)
    for mp in meshes:
        print('  %-30s material %-22s %6d verts %6d tris  uv %s  texture %s' % (
            mp.name[:30], mp.material[:22], len(mp.pos), len(mp.tris), mp.uv is not None, os.path.basename(mp.texture or '-')))
    t = np.concatenate([mp.pos[mp.tris] for mp in meshes]).astype(np.float64)
    print(render(t, a.out, a.yaw, a.size, a.pitch))


def cmd_swap(a):
    from swap import swap
    off = tuple(float(x) for x in a.offset.split(',')) if a.offset else (0, 0, 0)
    slots = [int(x) for x in a.slots.split(',')] if a.slots else None
    import json
    profile = json.load(open(a.profile, encoding='utf-8')) if a.profile else None
    plan = swap(a.romfs, a.model, a.file, a.mod, a.scale, a.rotate_y, off, slots, not a.no_textures, profile=profile)
    if a.preview:
        print(render(plan.triangles(a.pose), a.preview, a.yaw, a.size, a.pitch))


def cmd_backup(a):
    import backup
    md = a.mods
    if a.action == 'list':
        for mod, stamp, path, meta in backup.list_snapshots(md, a.mod):
            print('%-18s %-18s %4s files  %s' % (mod, stamp, meta.get('files', '?'), meta.get('reason', '')))
        print('mods:', ', '.join('%s (%s)' % (n, 'on' if on else 'off') for n, on in backup.mods(md)) or 'none')
    elif a.action == 'create':
        print(backup.snapshot(md, a.mod, 'by hand') or 'nothing to back up')
    elif a.action == 'restore':
        snaps = [s for s in backup.list_snapshots(md, a.mod) if not a.stamp or s[1] == a.stamp]
        if not snaps:
            raise SystemExit('no such backup')
        print('restored to', backup.restore(md, snaps[0][2]))
    elif a.action in ('on', 'off'):
        print(backup.set_enabled(md, a.mod, a.action == 'on'))
    elif a.action == 'remove':
        print('removed; backup kept at', backup.remove(md, a.mod))


def cmd_ui(a):
    import ui
    ui.main()


def main(argv=None):
    ap = argparse.ArgumentParser(prog='assetswap', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('inspect'); p.add_argument('romfs'); p.add_argument('model'); p.add_argument('--bones', action='store_true')
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser('export-obj'); p.add_argument('romfs'); p.add_argument('model'); p.add_argument('out')
    p.set_defaults(fn=cmd_export_obj)
    p = sub.add_parser('preview'); p.add_argument('romfs'); p.add_argument('model'); p.add_argument('out')
    p.add_argument('--yaw', type=float, default=0.0); p.add_argument('--size', type=int, default=900); p.add_argument('--pitch', type=float, default=0.0)
    p.add_argument('--pose', action='store_true', help='bend arms, a knee and the head (skin weight check)')
    p.set_defaults(fn=cmd_preview)
    p = sub.add_parser('preview-import'); p.add_argument('file'); p.add_argument('out')
    p.add_argument('--yaw', type=float, default=0.0); p.add_argument('--size', type=int, default=900); p.add_argument('--pitch', type=float, default=0.0)
    p.set_defaults(fn=cmd_preview_import)
    p = sub.add_parser('swap', help='replace a character model with an imported one, as a mod')
    p.add_argument('romfs'); p.add_argument('model'); p.add_argument('file')
    p.add_argument('--mod', required=True, help='mod folder to write: <mod>/romfs/<model>')
    p.add_argument('--scale', type=float); p.add_argument('--rotate-y', type=float, default=0.0)
    p.add_argument('--offset', help='x,y,z in metres'); p.add_argument('--slots', help='shape indices to use, e.g. 0,18')
    p.add_argument('--preview'); p.add_argument('--yaw', type=float, default=0.0)
    p.add_argument('--size', type=int, default=900); p.add_argument('--pitch', type=float, default=0.0)
    p.add_argument('--pose', action='store_true'); p.add_argument('--no-textures', action='store_true')
    p.add_argument('--profile', help='JSON profile: bone_map, slots, texture_map, attachments, scale, ... (see swap.py)')
    p.set_defaults(fn=cmd_swap)
    p = sub.add_parser('backup', help='list, create and restore backups of a mod; switch it on or off; remove it')
    p.add_argument('action', choices=['list', 'create', 'restore', 'on', 'off', 'remove'])
    p.add_argument('--mods', required=True, help='the mods folder of the game'); p.add_argument('--mod'); p.add_argument('--stamp')
    p.set_defaults(fn=cmd_backup)
    p = sub.add_parser('ui', help='open the Asset Swapper window')
    p.set_defaults(fn=cmd_ui)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
