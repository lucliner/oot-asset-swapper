"""Geometry replacement for the game's character models (patches a BFRES; builds nothing from scratch).

The model keeps everything it has: skeleton, materials, the vertex layout of every shape and the relocation table.
What is rewritten is the geometry of the shapes ("slots") the caller supplies, and the fields tied to it. The other
shapes are emptied (one degenerate triangle) unless `keep_others`. Layout facts, all measured from the game's files:

  * the file is sections of the relocation table: 0 = metadata, 1 = index buffers, 2 = vertex buffers, 3/4 = buffer info
    (zeros the game fills in), then the table. Entries are listed by the section their pointers point INTO, so when the
    sections move, the pointer values listed under sections 1..4 are shifted; the entries themselves stay where they are.
  * a shape has 3 LOD meshes (0x38 bytes each, one array), each with its own index buffer (a submesh {offset, count} and a
    byte-size word) and a first-vertex base. Here every LOD shares one set of vertices and one index list.
  * the bounding spheres per skin bone are bone-local (centre xyz, radius); the 6 boxes per shape are (centre, extent)
    in some bone's space, set here to a box that cannot be culled.
  * a vertex's bone indices (`_i0`, `_i1`) are skeleton matrix indices; a shape's skin list holds the bones it uses.
"""
import struct
from dataclasses import dataclass, replace

import numpy as np

from bfres import Bfres, fmt_info

ALIGN_BUF = 0x40
ALIGN_SEC = 0x1000


@dataclass
class SlotMesh:
    """Geometry for one shape slot, bound to the game's skeleton."""
    pos: np.ndarray            # (n, 3) model space, bind pose
    tris: np.ndarray           # (m, 3)
    joints: np.ndarray         # (n, k) skeleton matrix indices (smooth)
    weights: np.ndarray        # (n, k)
    normal: np.ndarray = None  # (n, 3)
    uv: np.ndarray = None      # (n, 2), v down as the game stores it
    color: np.ndarray = None   # (n, 4) 0..1
    texture: str = None        # base-colour image of the imported part (used by the texture step, not by the writer)
    uv_float: bool = False     # the UVs leave 0..1: declare the slot's UV attribute as half-float (same 4 bytes per vertex)


def _align(n, a):
    return (n + a - 1) // a * a


# ---- encoding of one vertex attribute (the inverse of Bfres.read_attr) ----
def encode_attr(fmt, a):
    kind, n, norm = fmt_info(fmt)
    count = len(a)
    if kind == 'p1010102':
        a = np.asarray(a, np.float64)
        w = np.zeros(count, np.uint32)
        for k in range(3):
            w |= (np.clip(np.rint(a[:, k] * 511), -511, 511).astype(np.int64) & 0x3ff).astype(np.uint32) << (10 * k)
        w |= (np.asarray(a[:, 3], np.int64).astype(np.uint32) & 3) << 30
        return w.astype('<u4').tobytes(), 4
    size = int(kind[1])
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None]
    a = np.concatenate([a, np.zeros((count, max(0, n - a.shape[1])), a.dtype)], axis=1)[:, :n]
    if norm == 'unorm':
        q = np.rint(np.clip(a, 0, 1) * ((1 << (8 * size)) - 1))
    elif norm == 'snorm':
        q = np.rint(np.clip(a, -1, 1) * ((1 << (8 * size - 1)) - 1))
    elif norm == 'float':
        return a.astype('<' + kind).tobytes(), size * n
    else:
        q = a
        hi = (1 << (8 * size)) - 1
        if q.size and q.max() > hi:
            raise ValueError('a bone index %d does not fit this slot\'s %d-bit index attribute' % (q.max(), 8 * size))
    return q.astype('<' + kind).tobytes(), size * n


def _comp_count(fmt):
    return fmt_info(fmt)[1]


def _quantize_weights(w, n):
    """(rows, n) float weights summing to 1 -> exact uint8 rows summing to 255 (as unorm8 needs)."""
    q = np.rint(w * 255).astype(np.int64)
    diff = 255 - q.sum(1)
    q[np.arange(len(q)), np.argmax(q, 1)] += diff
    return np.clip(q, 0, 255) / 255.0


def _normals(pos, tris):
    n = np.zeros_like(pos, np.float64)
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _tangents(pos, uv, tris, nrm):
    """Per-vertex tangent (xyz) and handedness (w = +1 or -1) from the UV layout."""
    t = np.zeros((len(pos), 3)); bt = np.zeros((len(pos), 3))
    if uv is not None:
        e1, e2 = pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]]
        d1, d2 = uv[tris[:, 1]] - uv[tris[:, 0]], uv[tris[:, 2]] - uv[tris[:, 0]]
        r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
        r = np.where(np.abs(r) < 1e-12, 1.0, 1.0 / np.where(np.abs(r) < 1e-12, 1.0, r))
        tt = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
        bb = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
        for k in range(3):
            np.add.at(t, tris[:, k], tt); np.add.at(bt, tris[:, k], bb)
    t = t - nrm * (nrm * t).sum(1, keepdims=True)
    bad = np.linalg.norm(t, axis=1) < 1e-9
    if bad.any():                                  # no usable UVs: any vector across the normal
        ref = np.where(np.abs(nrm[bad, 1:2]) < 0.9, np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))
        t[bad] = np.cross(ref, nrm[bad])
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    w = np.where((np.cross(nrm, t) * bt).sum(1) < 0, -1.0, 1.0)
    return t, w


def _pack_joints(sl, shape, matrix_to_bone):
    """Fit joints/weights to the slot's attribute capacity: the strongest K influences, renormalised."""
    K = sum(_comp_count(a.fmt) for a in shape.attrs if a.name in ('_i0', '_i1'))
    j, w = np.asarray(sl.joints), np.asarray(sl.weights, np.float64)
    order = np.argsort(-w, axis=1)[:, :K]
    j = np.take_along_axis(j, order, 1)
    w = np.take_along_axis(w, order, 1)
    if j.shape[1] < K:
        pad = K - j.shape[1]
        j = np.pad(j, ((0, 0), (0, pad))); w = np.pad(w, ((0, 0), (0, pad)))
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return j, w, K


def build_vertices(f, shape, sl):
    """The slot's vertex buffer bytes in the shape's own attribute layout."""
    n = len(sl.pos)
    nrm = np.asarray(sl.normal, np.float64) if sl.normal is not None else _normals(np.asarray(sl.pos, np.float64), sl.tris)
    uv = None if sl.uv is None else np.asarray(sl.uv, np.float64)
    tan, tw = _tangents(np.asarray(sl.pos, np.float64), uv, sl.tris, nrm)
    j, w, K = _pack_joints(sl, shape, f.matrix_to_bone)
    buf = np.zeros((n, shape.stride), np.uint8)
    for a in shape.attrs:
        c = _comp_count(a.fmt)
        if a.name == '_p0':
            v = np.asarray(sl.pos, np.float32)
        elif a.name == '_n0':
            v = np.c_[nrm, np.ones(n)]
        elif a.name == '_t0':
            v = np.c_[tan, np.where(tw < 0, 3, 1)]
        elif a.name.startswith('_u'):
            v = np.zeros((n, 2)) if (uv is None or a.name != '_u0') else uv
        elif a.name.startswith('_c'):
            v = np.ones((n, 4)) if sl.color is None else np.asarray(sl.color)
        elif a.name in ('_i0', '_i1'):
            lo = 0 if a.name == '_i0' else _comp_count(next(x.fmt for x in shape.attrs if x.name == '_i0'))
            v = j[:, lo:lo + c]
        elif a.name in ('_w0', '_w1'):
            continue                                # (filled below: both weight sets are quantised together)
        else:
            v = np.zeros((n, c))
        data, size = encode_attr(a.fmt, v)
        buf[:, a.offset:a.offset + size] = np.frombuffer(data, np.uint8).reshape(n, size)
    # weights across both sets must add to 1 together: re-quantise as one row
    w0 = next((a for a in shape.attrs if a.name == '_w0'), None)
    w1 = next((a for a in shape.attrs if a.name == '_w1'), None)
    if w0 is not None:
        c0 = _comp_count(w0.fmt); c1 = _comp_count(w1.fmt) if w1 is not None else 0
        q = _quantize_weights(w[:, :c0 + c1], c0 + c1) if c0 + c1 <= w.shape[1] else None
        if q is not None:
            d0, s0 = encode_attr(w0.fmt, q[:, :c0]); buf[:, w0.offset:w0.offset + s0] = np.frombuffer(d0, np.uint8).reshape(n, s0)
            if w1 is not None:
                d1, s1 = encode_attr(w1.fmt, q[:, c0:c0 + c1]); buf[:, w1.offset:w1.offset + s1] = np.frombuffer(d1, np.uint8).reshape(n, s1)
    return buf.tobytes(), j, w


def _degenerate(shape):
    return np.zeros((3, shape.stride), np.uint8).tobytes()


def patch_model(src, slots, keep_others=False, bone_positions=None, inv_bind=None):
    """Rewrite the geometry of a model file. `slots`: {shape index: SlotMesh}. Returns the new file bytes."""
    f = Bfres(src)
    b = bytearray(src[:f.sections[0][2]])           # metadata (section 0)
    old = [list(s) for s in f.sections]             # (ptr, pos, size, entry index, entry count)
    u64 = lambda o: struct.unpack_from('<Q', b, o)[0]
    put64 = lambda o, v: struct.pack_into('<Q', b, o, v)
    put32 = lambda o, v: struct.pack_into('<I', b, o, v)
    put16 = lambda o, v: struct.pack_into('<H', b, o, v)
    appended = bytearray()

    def append(data):
        nonlocal appended
        pos = _align(len(b) + len(appended), 8)
        appended += b'\0' * (pos - len(b) - len(appended)) + data
        return pos

    # pass 1: buffers
    inv_model = f.inv_model.copy() if f.inv_model is not None else None
    for m_, mat in (inv_bind or {}).items():              # inverse bind matrices of bones whose position changed
        inv_model[m_] = np.asarray(mat, np.float32).reshape(3, 4)
    idx_blob = bytearray(); vtx_blob = bytearray()
    placed = {}
    fmt_patch = []
    for s in f.shapes:
        if not s.vertex_count:
            continue                                # (an empty placeholder shape stays as it is)
        if s.index in slots:
            sl = slots[s.index]
            if sl.uv_float:
                for k, a in enumerate(s.attrs):
                    if a.name == '_u0' and fmt_info(a.fmt)[2] != 'float' and fmt_info(a.fmt)[0] in ('u2', 'i2'):
                        fmt_patch.append((u64(s.fvtx + 8) + k * 16 + 8, 0x1205))
                        s = replace(s, attrs=[replace(x, fmt=0x1205) if x.name == '_u0' else x for x in s.attrs])
            vb, j, w = build_vertices(f, s, sl)
            tris = np.asarray(sl.tris, np.int64)
            n = len(sl.pos)
            if n >= 0x10000:
                raise ValueError('shape %d: %d vertices; the index buffers here are 16-bit' % (s.index, n))
            ib = tris.astype('<u2').tobytes()
            count = tris.size
        elif keep_others:
            continue
        else:
            vb = _degenerate(s); ib = np.array([0, 1, 2], '<u2').tobytes(); count = 3; n = 3
            j = w = None
        ipos = _align(len(idx_blob), ALIGN_BUF); idx_blob += b'\0' * (ipos - len(idx_blob)) + ib
        vpos = _align(len(vtx_blob), ALIGN_BUF); vtx_blob += b'\0' * (vpos - len(vtx_blob)) + vb
        placed[s.index] = (ipos, len(ib), count, vpos, n, j, w)

    # pass 2: metadata (skin lists and bounding spheres are appended where a slot has real geometry)
    sec1_size = _align(len(idx_blob), 0x200)
    for s in f.shapes:
        if s.index not in placed:
            continue
        ipos, ilen, count, vpos, n, j, w = placed[s.index]
        # FVTX: vertex offset (relative to the pool; vertices follow the indices), size and count
        put32(s.fvtx + 0x48, sec1_size + vpos)
        put32(s.fvtx + 0x50, n)
        put32(u64(s.fvtx + 0x30), n * s.stride)
        # FSHP: the 3 LOD meshes share one index list and one vertex set
        mesh = u64(s.off + 0x18)
        for lod in range(3):
            q = mesh + lod * 0x38
            sub = u64(q)
            put32(sub, 0); put32(sub + 4, count)
            put32(u64(q + 0x18), ilen)
            put32(q + 0x20, ipos); put32(q + 0x2c, count); put32(q + 0x30, 0)
        if j is not None:
            used = sorted({int(x) for x in np.unique(j[w > 0])})
            bones = sorted({f.matrix_to_bone[m_] for m_ in used})
            pos = np.asarray(slots[s.index].pos, np.float64)
            sph = np.zeros((len(bones), 4), np.float32)
            for k, bn in enumerate(bones):
                m_ = f.bones[bn].smooth
                sel = ((j == m_) & (w > 0)).any(1)
                if m_ < 0 or not sel.any():
                    continue
                inv = np.eye(4); inv[:3, :] = inv_model[m_]
                loc = (np.c_[pos[sel], np.ones(sel.sum())] @ inv.T)[:, :3]
                c = (loc.min(0) + loc.max(0)) / 2
                sph[k, :3] = c
                sph[k, 3] = np.linalg.norm(loc - c, axis=1).max() * 1.05 + 0.005
            put64(s.off + 0x20, append(np.array(bones, '<u2').tobytes()))
            put64(s.off + 0x40, append(sph.astype('<f4').tobytes()))
            put16(s.off + 0x58, len(bones))
        bb = u64(s.off + 0x38)
        for e in range(6):
            struct.pack_into('<6f', b, bb + e * 24, 0, 0, 0, 1000, 1000, 1000)    # (centre, extent): never culled

    for loc, fmt in fmt_patch:
        put16(loc, fmt)
    for m_, mat in (inv_bind or {}).items():
        struct.pack_into('<12f', b, f.inv_off + m_ * 48, *np.asarray(mat, np.float32).reshape(-1))
    for bone, pos in (bone_positions or {}).items():      # local position of a bone (88-byte records; position after scale+rotation)
        struct.pack_into('<3f', b, f.bone_arr + bone * 0x58 + 0x4c, *pos)
    sec0_size = len(b) + len(appended)
    b += appended
    pool = max(old[1][1], _align(sec0_size, ALIGN_SEC))
    sec2_size = _align(len(vtx_blob), 0x200)
    new = [list(s) for s in old]
    new[0][2] = sec0_size
    new[1][1], new[1][2] = pool, sec1_size
    new[2][1], new[2][2] = pool + sec1_size, sec2_size
    end2 = _align(new[2][1] + sec2_size, ALIGN_SEC)
    new[2][2] = end2 - new[2][1]
    new[3][1] = end2
    new[4][1] = end2
    new[5][1] = end2 + old[4][2]

    # pointers listed under sections 1..4 move with their sections
    rlt = bytearray(src[f.rlt_off:])
    table = 16 + 24 * len(old)
    for k in range(1, 5):
        delta = new[k][1] - old[k][1]
        if not delta:
            continue
        for e in range(old[k][3], old[k][3] + old[k][4]):
            pos, cnt, noff, pad = struct.unpack_from('<IHBB', src, f.rlt_off + table + e * 8)
            for st in range(cnt):
                for o in range(noff):
                    loc = pos + (st * (noff + pad) + o) * 8
                    put64(loc, u64(loc) + delta)
    rlt_off = new[5][1]
    for k, s in enumerate(new):
        struct.pack_into('<QIIII', rlt, 16 + 24 * k, s[0], s[1], s[2], s[3], s[4])
    struct.pack_into('<I', rlt, 4, rlt_off)

    out = bytearray(b)
    out += b'\0' * (pool - len(out))
    out += idx_blob + b'\0' * (sec1_size - len(idx_blob))
    out += vtx_blob + b'\0' * (new[2][2] - len(vtx_blob))
    out += src[old[4][1]:old[4][1] + old[4][2]]                       # section 4 (buffer info: zeros the game fills)
    assert len(out) == rlt_off, (len(out), rlt_off)
    out += rlt
    struct.pack_into('<I', out, 0x18, rlt_off)
    struct.pack_into('<I', out, 0x1c, len(out))
    return bytes(out)
