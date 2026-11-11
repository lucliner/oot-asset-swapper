"""Reader for the game's character models (Switch BFRES, the `FRES` files in romfs `Model/`).

Reads the first model of a file: skeleton, shapes (with their meshes and vertex buffers) and material names.
Layouts were worked out from the files themselves and are checked against them by `export_obj` (tools/assetswap).
All offsets are absolute positions in the decompressed file. Vertex and index data sit in a data pool whose base is
the start of the second relocation-table section.
"""
import struct
from dataclasses import dataclass, field

import numpy as np


# ---- attribute formats: (type << 8 | kind) as stored in the FVTX attribute table ----
# kind: 1 unorm, 2 snorm, 3 uint, 4 sint, 5 float. type: component width and count (checked against the attribute
# offsets and strides of every character model: 0x15 is 16 bits x 4, 0x0b is 8 bits x 4, 0x18 is 32 bits x 3, ...)
_TYPE = {0x02: (1, 1), 0x09: (1, 2), 0x0b: (1, 4),          # (bytes per component, components)
         0x0a: (2, 1), 0x12: (2, 2), 0x15: (2, 4),
         0x14: (4, 1), 0x17: (4, 2), 0x18: (4, 3), 0x19: (4, 4)}
_PACKED = 0x0e02                                              # 10_10_10_2 snorm (normals, tangents)


def fmt_info(fmt):
    """(dtype, components, normalisation) of an attribute format, or None when it is not known."""
    if fmt == _PACKED:
        return 'p1010102', 4, 'snorm'
    t = _TYPE.get(fmt >> 8)
    if t is None or fmt & 0xff not in (1, 2, 3, 4, 5):
        return None
    size, n = t
    k = fmt & 0xff
    if k == 5:
        dt = {2: 'f2', 4: 'f4'}.get(size)
    else:
        dt = ('i' if k in (2, 4) else 'u') + str(size)
    if dt is None:
        return None
    return dt, n, {1: 'unorm', 2: 'snorm', 3: 'uint', 4: 'sint', 5: 'float'}[k]


@dataclass
class Bone:
    index: int
    name: str
    parent: int
    smooth: int            # smooth-skin matrix index (-1: none)
    rigid: int
    flags: int
    scale: tuple
    rot: tuple             # euler XYZ (radians) or quaternion, see flags
    pos: tuple


@dataclass
class Attr:
    name: str
    fmt: int
    offset: int
    buffer: int


@dataclass
class Shape:
    index: int
    name: str
    material: int
    bone: int
    skin_bone_count: int
    skin_count: int        # influences per vertex: 0 rigid, 1 rigid skin, 2..8 smooth
    vertex_count: int
    attrs: list
    stride: int
    vb_off: int            # absolute offset of the vertex buffer
    index_off: int         # absolute offset of the index buffer
    index_count: int
    index_fmt: int         # 0 u8, 1 u16, 2 u32
    prim: int
    skin_bones: list = field(default_factory=list)   # bones the shape uses (bone indices, for the GPU)
    off: int = 0           # offset of its FSHP
    fvtx: int = 0          # offset of its FVTX


class Bfres:
    def __init__(self, data):
        self.b = data
        b = data
        if b[:4] != b'FRES':
            raise ValueError('not a BFRES file')
        self.version = struct.unpack_from('<I', b, 8)[0]
        self.size = struct.unpack_from('<I', b, 0x1c)[0]
        self._rlt()
        self.name = self._str(struct.unpack_from('<Q', b, 0x20)[0])
        mdl_array = self._u64(0x28)
        mdl_dict = self._u64(0x30)
        self.model_count = self._dict_count(mdl_dict)
        if self.model_count < 1:
            raise ValueError('no model in this file')
        self.model_off = mdl_array
        self._model(mdl_array)

    # -- low level
    def _u64(self, o): return struct.unpack_from('<Q', self.b, o)[0]
    def _u32(self, o): return struct.unpack_from('<I', self.b, o)[0]
    def _u16(self, o): return struct.unpack_from('<H', self.b, o)[0]

    def _str(self, o):
        if not o: return ''
        n = self._u16(o)
        return self.b[o + 2:o + 2 + n].decode('utf-8', 'replace')

    def _dict_count(self, o):
        if not o: return 0
        # (this version's dictionaries have no magic: a signature word, then the entry count)
        return struct.unpack_from('<i', self.b, o + 4)[0]

    def _rlt(self):
        i = self._u32(0x18)
        if self.b[i:i + 4] != b'_RLT':
            raise ValueError('relocation table not found')
        cnt = self._u32(i + 8)
        self.sections = [struct.unpack_from('<QIIII', self.b, i + 16 + s * 24) for s in range(cnt)]
        # section 1 holds the index buffers, section 2 the vertex buffers; offsets in shapes count from section 1
        self.rlt_off = i
        self.pool = self.sections[1][1] if cnt > 1 else 0

    # -- model
    def _model(self, o):
        b = self.b
        if b[o:o + 4] != b'FMDL':
            raise ValueError('FMDL expected at 0x%x' % o)
        self.model_name = self._str(self._u64(o + 8))
        skl = self._u64(o + 0x18)
        vb_arr, shp_arr, mat_arr = self._u64(o + 0x20), self._u64(o + 0x28), self._u64(o + 0x38)
        nvb, nshp, nmat, _nud = struct.unpack_from('<4H', b, o + 0x68)
        self.bones = self._skeleton(skl)
        self.materials = [self._str(self._u64(mat_arr + i * 0xb0 + 8)) for i in range(nmat)]
        self.shape_arr = shp_arr
        self.shapes = [self._shape(shp_arr + i * 0x60, vb_arr, i) for i in range(nshp)]

    def _skeleton(self, o):
        b = self.b
        if b[o:o + 4] != b'FSKL':
            raise ValueError('FSKL expected at 0x%x' % o)
        bone_arr, mat2bone, inv = self._u64(o + 0x10), self._u64(o + 0x18), self._u64(o + 0x20)
        self.bone_arr = bone_arr
        self.inv_off = inv
        nbone, nsmooth, nrigid = struct.unpack_from('<3H', b, o + 0x38)
        self.skeleton_flags = self._u32(o + 4)
        self.matrix_to_bone = list(struct.unpack_from('<%dH' % (nsmooth + nrigid), b, mat2bone))
        self.smooth_count, self.rigid_count = nsmooth, nrigid
        self.inv_model = (np.frombuffer(b, '<f4', (nsmooth + nrigid) * 12, inv).reshape(-1, 3, 4)
                          if inv else None)
        bones = []
        for i in range(nbone):
            p = bone_arr + i * 0x58
            idx, par, sm, rg, _bb, _ud = struct.unpack_from('<HhhhHH', b, p + 0x20)
            fl = self._u32(p + 0x2c)
            f = struct.unpack_from('<10f', b, p + 0x30)
            bones.append(Bone(idx, self._str(self._u64(p)), par, sm, rg, fl, f[0:3], f[3:7], f[7:10]))
        return bones

    def _shape(self, o, vb_arr, i):
        b = self.b
        if b[o:o + 4] != b'FSHP':
            raise ValueError('FSHP expected at 0x%x' % o)
        name = self._str(self._u64(o + 8))
        vb = self._u64(o + 0x10)
        mesh, skin_arr = self._u64(o + 0x18), self._u64(o + 0x20)
        _idx, mat, bone, _vbi, nskinb, skin, _nt = struct.unpack_from('<5H2B', b, o + 0x50)
        # FVTX (some equipment models hold a placeholder shape whose vertex buffer is all zeros: no geometry)
        if b[vb:vb + 4] == b'\x00\x00\x00\x00':
            return Shape(i, name, mat, bone, nskinb, skin, 0, [], 0, 0, 0, 0, 0, 0, off=o, fvtx=vb)
        if b[vb:vb + 4] != b'FVTX':
            raise ValueError('FVTX expected at 0x%x' % vb)
        attr_arr = self._u64(vb + 8)
        size_arr, stride_arr = self._u64(vb + 0x30), self._u64(vb + 0x38)
        vb_off = self._u32(vb + 0x48)
        nattr, nbuf = b[vb + 0x4c], b[vb + 0x4d]
        vcount = self._u32(vb + 0x50)
        attrs = []
        for k in range(nattr):
            p = attr_arr + k * 16
            nm = self._str(self._u64(p))
            fmt, _pad, off, buf = struct.unpack_from('<4H', b, p + 8)
            attrs.append(Attr(nm, fmt, off, buf))
        if nbuf != 1:
            raise NotImplementedError('shape %s: %d vertex buffers (only 1 is supported so far)' % (name, nbuf))
        stride = self._u32(stride_arr)
        # mesh (LOD 0 only)
        face_off, prim, ifmt, icount = struct.unpack_from('<4I', b, mesh + 0x20)
        skin_bones = list(struct.unpack_from('<%dH' % nskinb, b, skin_arr)) if skin_arr and nskinb else []
        return Shape(i, name, mat, bone, nskinb, skin, vcount, attrs, stride,
                     self.pool + vb_off, self.pool + face_off, icount, ifmt, prim, skin_bones, o, vb)

    # -- geometry
    def attr(self, shape, name):
        return next((a for a in shape.attrs if a.name == name), None)

    def read_attr(self, shape, name):
        """Decode one vertex attribute of a shape to a float32 (or int) array of shape (vertices, components)."""
        a = self.attr(shape, name)
        if a is None:
            return None
        info = fmt_info(a.fmt)
        if info is None:
            raise NotImplementedError('attribute format 0x%04x (%s)' % (a.fmt, name))
        kind, n, norm = info
        raw = np.frombuffer(self.b, np.uint8, shape.vertex_count * shape.stride, shape.vb_off)
        raw = raw.reshape(shape.vertex_count, shape.stride)
        if kind == 'p1010102':
            w = raw[:, a.offset:a.offset + 4].copy().view('<u4')[:, 0].astype(np.int64)
            out = np.empty((shape.vertex_count, 4), np.float32)
            for k in range(3):
                v = (w >> (10 * k)) & 0x3ff
                v = np.where(v & 0x200, v - 0x400, v)
                out[:, k] = np.maximum(v / 511.0, -1.0)
            out[:, 3] = ((w >> 30) & 3).astype(np.float32)
            return out
        size = int(kind[1])
        arr = raw[:, a.offset:a.offset + size * n].copy().view('<' + kind)
        if norm == 'unorm':
            return arr.astype(np.float32) / float((1 << (8 * size)) - 1)
        if norm == 'snorm':
            return np.maximum(arr.astype(np.float32) / float((1 << (8 * size - 1)) - 1), -1.0)
        return arr.astype(np.float32) if norm == 'float' else arr

    def indices(self, shape):
        dt = {0: '<u1', 1: '<u2', 2: '<u4'}[shape.index_fmt]
        return np.frombuffer(self.b, dt, shape.index_count, shape.index_off).astype(np.int64)

    def bone_of_matrix(self, matrix_index):
        """Skeleton bone for a vertex's bone index. Vertex indices (`_i0`, `_i1`) are skeleton matrix indices (smooth
        ones first, then rigid); the shape's own `skin_bones` list is only the bones it uses, as bone indices."""
        return self.matrix_to_bone[int(matrix_index)]
