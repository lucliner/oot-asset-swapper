"""The game's models with their own textures, for previews: every shape's base-colour texture decoded from the Tex folder.

Needs `texture2ddecoder` (ASTC and BCn); without it models are drawn untextured.
"""
import numpy as np
from PIL import Image

import bntx
import textures
from bfres import Bfres
from skeleton import world_matrices

try:
    import texture2ddecoder as _t2d
except Exception:
    _t2d = None

ASTC = {0x2d: (4, 4), 0x2e: (5, 4), 0x2f: (5, 5), 0x30: (6, 5), 0x31: (6, 6), 0x32: (8, 5), 0x33: (8, 6), 0x34: (8, 8),
        0x35: (10, 5), 0x36: (10, 6), 0x37: (10, 8), 0x38: (10, 10), 0x39: (12, 10), 0x3a: (12, 12)}
BCN = {0x1a: ('bc1', 8), 0x1b: ('bc2', 16), 0x1c: ('bc3', 16), 0x1d: ('bc4', 8), 0x1e: ('bc5', 16), 0x1f: ('bc6', 16), 0x20: ('bc7', 16)}
_cache = {}


def decode(data, max_side=256, rgba=False):
    """RGB uint8 (h, w, 3) (RGBA with `rgba`) of the base level of a one-texture BNTX file's bytes; None for a format this cannot
    read."""
    if _t2d is None:
        return None
    d = bntx.Brti(data)
    t = d.fmt >> 8
    bh = 1 << d.block_height_log2
    if t in ASTC:
        bw, bhp = ASTC[t]
        gw, gh = -(-d.w // bw), -(-d.h // bhp)
        blk = bntx.deswizzle(data[bntx.DATA_START:], gw, gh, 16, bh)
        px = _t2d.decode_astc(blk.tobytes(), gw * bw, gh * bhp, bw, bhp)
        w, h = gw * bw, gh * bhp
    elif t in BCN:
        name, bpp = BCN[t]
        gw, gh = -(-d.w // 4), -(-d.h // 4)
        blk = bntx.deswizzle(data[bntx.DATA_START:], gw, gh, bpp, bh)
        fn = getattr(_t2d, 'decode_' + name, None)
        if fn is None:
            return None
        px = fn(blk.tobytes(), gw * 4, gh * 4)
        w, h = gw * 4, gh * 4
    else:
        return None
    a = np.frombuffer(px, np.uint8).reshape(h, w, 4)[..., [2, 1, 0, 3] if rgba else [2, 1, 0]]
    im = Image.fromarray(np.ascontiguousarray(a[:d.h, :d.w]))
    if max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    return np.asarray(im, np.uint8)


def albedo(rom, f, material, max_side=256):
    """The decoded base-colour texture of a material (None when it has none, or it cannot be decoded)."""
    names = [n for n in textures.material_textures(f, material) if '_Alb' in n or n.endswith('Alb')]
    names.sort(key=lambda n: 0 if 'Sclera' in n else 1)              # an eye has an iris and a sclera map: the white one reads as an eye
    for n in names:
        if True:
            key = (n, max_side)
            if key not in _cache:
                try:
                    _cache[key] = decode(rom.read('Tex/%s.bntx.zs' % n), max_side)
                except Exception:
                    _cache[key] = None
                if len(_cache) > 600:
                    _cache.pop(next(iter(_cache)))
            return _cache[key]
    return None


def textured_parts(rom, rel, max_side=256):
    """[{'tris': (n, 3, 3), 'uv': (n, 3, 2) or None, 'tex': (h, w, 3) uint8 or None}] for a game model in its bind pose, plus
    {'tris', 'height', 'bones', 'shapes'}."""
    f = Bfres(rom.read(rel))
    W = world_matrices(f)
    parts, total = [], 0
    for sh in f.shapes:
        if not sh.vertex_count:
            continue
        pos = f.read_attr(sh, '_p0')[:, :3].astype(np.float64)
        if sh.skin_count <= 1:
            h = np.c_[pos, np.ones(len(pos))]
            if sh.skin_count == 0 or not sh.skin_bones:
                pos = (h @ W[sh.bone].T)[:, :3]
            else:
                vb = f.read_attr(sh, '_i0')[:, 0].astype(int)
                q = np.empty_like(pos)
                for k in np.unique(vb):
                    sel = vb == k
                    q[sel] = (h[sel] @ W[f.matrix_to_bone[k]].T)[:, :3]
                pos = q
        idx = f.indices(sh).reshape(-1, 3)
        total += len(idx)
        uv = None
        try:
            u = f.read_attr(sh, '_u0')
            if u is not None and len(u) == len(pos):
                uv = np.asarray(u[:, :2], np.float64)[idx]
        except Exception:
            uv = None
        tex = albedo(rom, f, sh.material, max_side) if uv is not None else None
        fn = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
        vn = np.zeros_like(pos)
        for k in range(3):
            np.add.at(vn, idx[:, k], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        parts.append({'tris': pos[idx], 'uv': uv, 'tex': tex, 'nrm': vn[idx].astype(np.float32)})
    pts = np.concatenate([p['tris'].reshape(-1, 3) for p in parts]) if parts else np.zeros((1, 3))
    facts = {'tris': int(total), 'height': float(np.ptp(pts[:, 1])), 'bones': len(f.bones), 'shapes': len(parts)}
    return parts, facts
