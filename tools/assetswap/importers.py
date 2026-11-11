"""Readers for models from other games. Each gives a list of `Mesh` parts: one material each, positions in metres
(+y up, +z forward of the model as stored: the caller orients it), triangles as indices.

OBJ is read here (static, unrigged: the weights come from auto-skinning). glTF/GLB, which can carry a skeleton, comes next.
"""
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    name: str
    material: str
    pos: np.ndarray                 # (n, 3) float32
    tris: np.ndarray                # (m, 3) int64
    uv: np.ndarray = None           # (n, 2) float32, v up as in OBJ
    normal: np.ndarray = None       # (n, 3) float32
    texture: str = None             # path of the base-colour image, when the material names one
    joints: np.ndarray = None       # (n, k) int, source joints (rigged formats)
    weights: np.ndarray = None      # (n, k) float32
    extra: dict = field(default_factory=dict)


def load_obj(path):
    """Wavefront OBJ (+ MTL for texture names) -> [Mesh]. Faces of any size are fan-triangulated; vertices are split
    where position, texture coordinate or normal differ, so each Mesh has one index per drawn corner."""
    base = os.path.dirname(os.path.abspath(path))
    V, T, N = [], [], []
    mtllib = {}
    parts = []                       # (name, material, [corner tuples per face])
    cur_o, cur_m = 'mesh', ''
    cur = None

    def start():
        nonlocal cur
        cur = [cur_o, cur_m, []]
        parts.append(cur)

    def corner(tok):
        a = tok.split('/')
        def idx(s, n):
            if not s: return -1
            i = int(s)
            return i - 1 if i > 0 else n + i
        return (idx(a[0], len(V)), idx(a[1], len(T)) if len(a) > 1 else -1, idx(a[2], len(N)) if len(a) > 2 else -1)

    for line in open(path, encoding='utf-8', errors='replace'):
        t = line.split()
        if not t or t[0].startswith('#'):
            continue
        k = t[0]
        if k == 'v': V.append([float(x) for x in t[1:4]])
        elif k == 'vt': T.append([float(x) for x in t[1:3]])
        elif k == 'vn': N.append([float(x) for x in t[1:4]])
        elif k in ('o', 'g'):
            cur_o = ' '.join(t[1:]) or 'mesh'
            cur = None
        elif k == 'usemtl':
            cur_m = ' '.join(t[1:])
            cur = None
        elif k == 'mtllib':
            mtllib = _read_mtl(os.path.join(base, ' '.join(t[1:])))
        elif k == 'f':
            if cur is None: start()
            c = [corner(x) for x in t[1:]]
            for i in range(1, len(c) - 1):
                cur[2].append((c[0], c[i], c[i + 1]))
    V = np.array(V, np.float32).reshape(-1, 3)
    T = np.array(T, np.float32).reshape(-1, 2)
    N = np.array(N, np.float32).reshape(-1, 3)
    out = []
    for name, mat, faces in parts:
        if not faces:
            continue
        remap, corners, tris = {}, [], []
        for f in faces:
            tri = []
            for c in f:
                if c not in remap:
                    remap[c] = len(corners)
                    corners.append(c)
                tri.append(remap[c])
            tris.append(tri)
        pi = np.array([c[0] for c in corners]); ti = np.array([c[1] for c in corners]); ni = np.array([c[2] for c in corners])
        tex = mtllib.get(mat)
        out.append(Mesh(name, mat, V[pi],
                        np.array(tris, np.int64),
                        T[ti] if len(T) and (ti >= 0).all() else None,
                        N[ni] if len(N) and (ni >= 0).all() else None,
                        os.path.join(base, tex) if tex else None))
    return out


def _read_mtl(path):
    out, cur = {}, None
    if not os.path.exists(path):
        return out
    for line in open(path, encoding='utf-8', errors='replace'):
        t = line.split(None, 1)
        if len(t) < 2: continue
        if t[0] == 'newmtl': cur = t[1].strip()
        elif t[0] == 'map_Kd' and cur: out[cur] = t[1].strip().split()[-1]
    return out
