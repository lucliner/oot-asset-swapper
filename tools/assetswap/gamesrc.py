"""A model from the game itself as the model to put in: its shapes become the imported parts, its skeleton the source rig.

The source is named `game:Model/<file>.bfres.zs`. Skin weights come from the game's own vertex data, so the animations of the
model it replaces drive it through the usual retargeting (joints with the same name map to each other).
"""
import os
import tempfile

import numpy as np
from PIL import Image

import gametex
from bfres import Bfres
from gltf_io import Rig
from importers import Mesh
from skeleton import world_matrices

PREFIX = 'game:'


def is_game(path):
    return str(path).startswith(PREFIX)


def load(rom, path, cache_dir=None):
    """([Mesh], Rig) of the game model named by `game:<romfs path>`."""
    rel = path[len(PREFIX):]
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'assetswap_cache', 'game')
    os.makedirs(cache_dir, exist_ok=True)
    f = Bfres(rom.read(rel))
    W = world_matrices(f)
    stem = os.path.basename(rel).split('.')[0]
    rig = Rig([b.name for b in f.bones], [b.parent for b in f.bones], np.asarray(W, np.float64))
    meshes = []
    for sh in f.shapes:
        if sh.vertex_count <= 3:
            continue                                                   # (a placeholder shape)
        pos = f.read_attr(sh, '_p0')[:, :3].astype(np.float64)
        n = len(pos)
        joints = np.zeros((n, 4), np.int64)
        weights = np.zeros((n, 4), np.float32)
        if sh.skin_count > 1:
            i0 = np.asarray(f.read_attr(sh, '_i0'), np.int64)
            w0 = np.asarray(f.read_attr(sh, '_w0'), np.float32)
            k = min(4, i0.shape[1], w0.shape[1] + 1)
            for c in range(min(4, i0.shape[1])):
                joints[:, c] = [f.matrix_to_bone[m] if m < len(f.matrix_to_bone) else 0 for m in i0[:, c]]
            if w0.shape[1] >= 1:
                weights[:, :w0.shape[1]] = w0[:, :4]
            if i0.shape[1] > w0.shape[1]:                                   # (the last weight is what the others leave)
                weights[:, w0.shape[1]] = np.clip(1.0 - w0.sum(1), 0, 1) if w0.shape[1] < 4 else 0
            s = weights.sum(1, keepdims=True)
            weights = np.where(s > 1e-6, weights / np.maximum(s, 1e-6), weights)
        else:
            h = np.c_[pos, np.ones(n)]
            if sh.skin_count == 0 or not sh.skin_bones:
                bone = np.full(n, sh.bone)
            else:
                vb = f.read_attr(sh, '_i0')[:, 0].astype(int)
                bone = np.array([f.matrix_to_bone[m] for m in vb])
            for b in np.unique(bone):
                sel = bone == b
                pos[sel] = (h[sel] @ W[b].T)[:, :3]
            joints[:, 0] = bone
            weights[:, 0] = 1.0
        idx = f.indices(sh).reshape(-1, 3).astype(np.int64)
        uv = None
        try:
            u = f.read_attr(sh, '_u0')
            if u is not None and len(u) == n:
                uv = np.asarray(u[:, :2], np.float32)                      # (v up, as the importers keep it)
        except Exception:
            uv = None
        nrm = None
        try:
            nn = f.read_attr(sh, '_n0')
            if nn is not None and len(nn) == n:
                nrm = np.asarray(nn[:, :3], np.float32)
        except Exception:
            nrm = None
        mat = f.materials[sh.material]
        tex = None
        if uv is not None:
            im = gametex.albedo(rom, f, sh.material, 1024)
            if im is not None:
                tex = os.path.join(cache_dir, '%s_%s.png' % (stem, mat))
                if not os.path.exists(tex):
                    Image.fromarray(im).save(tex)
        meshes.append(Mesh(sh.name, mat, pos.astype(np.float32), idx, uv, nrm, tex, joints, weights))
    return meshes, rig
