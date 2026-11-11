"""Fit an unrigged mesh to one of the game's skeletons and give it skin weights.

The mesh is assumed to be in a T-pose or A-pose facing +z with +y up (rotate it first when it is not). Weights come from
the distance to the body bones of the target (spine, neck, head, clavicles, arms, wrists, waist, legs, ankles, toes):
cloth, hair, finger and face bones are left out, since the game drives those by physics or by facial animation.
"""
import re

import numpy as np

CORE = re.compile(r'^(Spine_\d|Waist|Neck|Head(_2)?|Clavicle_[LR]|Arm_[12]_[LR]|Wrist_[LR]|Leg_[12]_[LR]|Ankle_[LR]|Toe_[LR])$')
# how far a bone with no core child reaches past its joint (metres, along the parent's direction): the end of a limb
# must catch geometry that is longer than the target's own limb (a wing, a tail of a sleeve)
REACH = {'Wrist': 2.0, 'Head_2': 0.6, 'Toe': 0.4, 'Ankle': 0.2}


def core_bones(f):
    return [b for b in f.bones if CORE.match(b.name) and b.smooth >= 0]


def segments(f, W):
    """(bones, start (k,3), end (k,3)) in model space: one segment per core bone."""
    bones = core_bones(f)
    ids = {b.index for b in bones}
    pos = lambda i: W[i][:3, 3]
    kids = {i: [] for i in ids}
    for b in bones:
        p = b.parent
        while p >= 0 and p not in ids:
            p = f.bones[p].parent
        if p in kids:
            kids[p].append(b.index)
    A, B = [], []
    for b in bones:
        a = pos(b.index)
        if kids[b.index]:
            e = np.mean([pos(k) for k in kids[b.index]], axis=0)
            if np.linalg.norm(e - a) < 1e-4:
                e = a + np.array([0, 1e-3, 0])
        else:
            p = b.parent
            d = a - pos(p) if p >= 0 else np.array([0, 1.0, 0])
            if np.linalg.norm(d) < 1e-6:
                d = np.array([0, 1.0, 0])
            d /= np.linalg.norm(d)
            reach = next((v for k, v in REACH.items() if b.name.startswith(k)), 0.1)
            if b.name.startswith('Toe'):
                d = np.array([0, 0, 1.0])
            e = a + d * reach
        A.append(a); B.append(e)
    return bones, np.array(A), np.array(B)


def seg_distance(P, A, B):
    """Distance from each point (n,3) to each segment (k,3)-(k,3): (n, k)."""
    ab = B - A
    t = ((P[:, None, :] - A[None]) * ab[None]).sum(2) / np.maximum((ab * ab).sum(1), 1e-12)[None]
    t = np.clip(t, 0, 1)
    c = A[None] + t[..., None] * ab[None]
    return np.linalg.norm(P[:, None, :] - c, axis=2)


def fit(meshes, f, W, scale=None, rotate_y=0.0, offset=(0.0, 0.0, 0.0), target_height=None):
    """Transform of the imported meshes into the target's space: returns (scale, R (3,3), translation (3,)) so that
    p' = scale * R @ p + t puts the mesh on the ground (y = 0) at the target's height, centred on x and z of its origin."""
    ry = np.radians(rotate_y)
    R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    P = np.concatenate([m.pos for m in meshes]).astype(np.float64) @ R.T
    lo, hi = P.min(0), P.max(0)
    if scale is None:
        scale = target_height / max(hi[1] - lo[1], 1e-6)
    t = np.array([-(lo[0] + hi[0]) / 2 * scale, -lo[1] * scale, 0.0]) + np.array(offset)
    return scale, R, t


def autoskin(f, W, pos, tris, k=4, sigma=0.04, smooth=6):
    """Skin weights for one mesh in the target's bind pose: (joints (n,k) matrix indices, weights (n,k))."""
    bones, A, B = segments(f, W)
    pos = np.asarray(pos, np.float64)
    # merge coincident vertices so smoothing crosses UV seams
    key = np.round(pos * 1e5).astype(np.int64)
    uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    U = pos[first]
    d = seg_distance(U, A, B)
    w = np.exp(-(d - d.min(1, keepdims=True)) / sigma)
    w /= w.sum(1, keepdims=True)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(inv[e], axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    for _ in range(smooth):
        acc = np.zeros_like(w); cnt = np.zeros(len(w))
        np.add.at(acc, e[:, 0], w[e[:, 1]]); np.add.at(acc, e[:, 1], w[e[:, 0]])
        np.add.at(cnt, e[:, 0], 1); np.add.at(cnt, e[:, 1], 1)
        has = cnt > 0
        w[has] = 0.5 * w[has] + 0.5 * acc[has] / cnt[has, None]
    w = w[inv]
    top = np.argsort(-w, axis=1)[:, :k]
    wk = np.take_along_axis(w, top, 1)
    wk /= np.maximum(wk.sum(1, keepdims=True), 1e-12)
    mat = np.array([b.smooth for b in bones])
    return mat[top], wk.astype(np.float32)
