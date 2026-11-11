"""Pose a model with linear blend skinning, to check skin weights by eye (`preview --pose`)."""
import numpy as np

from skeleton import world_matrices

# bone name -> (world axis, degrees): the bone and everything below it turns about its own joint. Arms down, a knee
# bent, the head turned: enough to show whether a limb's geometry follows its own bones.
TEST_POSE = {
    'Arm_1_L': ((0, 0, 1), -55), 'Arm_1_R': ((0, 0, 1), 55),
    'Arm_2_L': ((0, 0, 1), -20), 'Arm_2_R': ((0, 0, 1), 20),
    'Leg_1_L': ((1, 0, 0), 45), 'Leg_2_L': ((1, 0, 0), -60),
    'Head': ((0, 1, 0), 40), 'Spine_2': ((1, 0, 0), 12),
}


def rot_about(point, axis, deg):
    a = np.radians(deg); x, y, z = np.array(axis, float) / np.linalg.norm(axis)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                  [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                  [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = point - R @ point
    return M


def pose_matrices(f, W, pose=TEST_POSE):
    """Per-bone world-space transform that moves a rest-pose model-space point to the posed one."""
    M = []
    for b in f.bones:
        D = rot_about(W[b.index][:3, 3], *pose[b.name]) if b.name in pose else np.eye(4)
        M.append(M[b.parent] @ D if b.parent >= 0 else D)
    return M


def posed_positions(f, shape, rest, M):
    """Linear blend skinning of a smooth-skinned shape's rest positions (n, 3)."""
    n = len(rest)
    h = np.c_[rest, np.ones(n)]
    out = np.zeros((n, 3))
    wsum = np.zeros(n)
    for iname, wname in (('_i0', '_w0'), ('_i1', '_w1')):
        I, Wt = f.read_attr(shape, iname), f.read_attr(shape, wname)
        if I is None or Wt is None:
            continue
        I = I.astype(int).reshape(n, -1); Wt = Wt.reshape(n, -1)
        for k in range(I.shape[1]):
            for m in np.unique(I[:, k]):
                sel = (I[:, k] == m) & (Wt[:, k] > 0)
                if not sel.any():
                    continue
                out[sel] += Wt[sel, k, None] * (h[sel] @ M[f.matrix_to_bone[m]].T)[:, :3]
                wsum[sel] += Wt[sel, k]
    rigid = wsum < 1e-6
    out[rigid] = rest[rigid]
    return out
