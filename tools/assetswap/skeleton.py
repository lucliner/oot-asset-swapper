"""Bind-pose maths of the game's skeletons."""
import numpy as np


def euler_matrix(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Z @ Y @ X


def world_matrices(m):
    """4x4 bind-pose world matrix of every bone (euler XYZ rotation, the mode every bone of the game's models uses)."""
    out = []
    for b in m.bones:
        if not b.flags & 0x1000:
            raise NotImplementedError('bone %s uses quaternion rotation' % b.name)
        L = np.eye(4)
        L[:3, :3] = euler_matrix(*b.rot[:3]) * np.array(b.scale)
        L[:3, 3] = b.pos
        out.append(out[b.parent] @ L if b.parent >= 0 else L)
    return out
