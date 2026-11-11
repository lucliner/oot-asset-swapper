"""Retarget a rigged model onto one of the game's skeletons.

1. `auto_map` guesses which game bone each source joint belongs to from the names (MMD, Mixamo, Unity/VRM, Blender and
   plain names); the user can correct any of it (`bone_map` in the swap profile: {source joint: game bone, or null}).
2. `resolve` gives every source joint a game bone: its own mapping, else its nearest mapped ancestor's.
3. `deform` moves the mesh from the source's rest pose into the game's bind pose: each game bone gets the transform that
   takes its source bone's start point and direction onto its own (stretched along the bone to the game bone's length),
   and a vertex blends those transforms by its source weights. The source's own weights then stay valid.
"""
import re

import numpy as np

# game bone -> names it goes by elsewhere (normalised: lower case, no separators); side is handled separately
ROLES = [
    ('Spine_1', ['upperbody', 'spine', 'spine1', 'abdomen', 'chest']),
    ('Spine_2', ['upperbody2', 'chest', 'spine2', 'spine1', 'upperchest', 'ribcage']),
    ('Waist', ['lowerbody', 'hips', 'pelvis', 'waist', 'hip']),
    ('Neck', ['neck', 'neck01', 'neck1']),
    ('Head', ['head']),
    ('Clavicle', ['shoulder', 'clavicle', 'collar']),
    ('Arm_1', ['arm', 'upperarm', 'upper_arm']),
    ('Arm_2', ['elbow', 'lowerarm', 'forearm', 'fore_arm']),
    ('Wrist', ['wrist', 'hand', 'handtwist']),
    ('Leg_1', ['leg', 'upperleg', 'thigh', 'upleg']),
    ('Leg_2', ['knee', 'lowerleg', 'calf', 'shin']),
    ('Ankle', ['ankle', 'foot']),
    ('Toe', ['toe', 'toes', 'toebase']),
]
SIDED = {'Clavicle', 'Arm_1', 'Arm_2', 'Wrist', 'Leg_1', 'Leg_2', 'Ankle', 'Toe'}
SKIP = re.compile(r'ik|end|finger|thumb|index|middle|ring|pinky|little|eye|brow|hair|skirt|tail|jaw|tongue|lip|cheek|'
                  r'twist\d|_d$|_ex$|dummy|shadow|cloth|cape|coat|ribbon|bust|breast', re.I)


# the child each body bone points at (its axis), so branching bones (spine, clavicles) get a meaningful direction
AXIS = {'Spine_1': 'Spine_2', 'Spine_2': 'Neck', 'Neck': 'Head',
        'Clavicle_L': 'Arm_1_L', 'Arm_1_L': 'Arm_2_L', 'Arm_2_L': 'Wrist_L',
        'Clavicle_R': 'Arm_1_R', 'Arm_1_R': 'Arm_2_R', 'Arm_2_R': 'Wrist_R',
        'Leg_1_L': 'Leg_2_L', 'Leg_2_L': 'Ankle_L', 'Leg_1_R': 'Leg_2_R', 'Leg_2_R': 'Ankle_R'}
NO_AXIS = {'Waist'}          # sits where the legs and the spine start: moved to its place, not turned or stretched
MIN_BONE = 0.03              # metres: a bone shorter than this has no usable direction


def _norm(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _side(name):
    n = name.lower()
    if re.search(r'(^|[_.\-\s])(l|left)($|[_.\-\s0-9])|left|\.l$|_l$', n):
        return 'L'
    if re.search(r'(^|[_.\-\s])(r|right)($|[_.\-\s0-9])|right|\.r$|_r$', n):
        return 'R'
    return None


def _role(name):
    base = re.sub(r'(^|[_.\-\s])(l|r|left|right)($|[_.\-\s])', ' ', name.lower())
    base = re.sub(r'(left|right)', '', base)
    n = _norm(base)
    best = None
    for role, aliases in ROLES:
        for rank, a in enumerate(aliases):
            if n == a:
                cand = (rank, role)
                if best is None or cand < best:
                    best = cand
    return best[1] if best else None


def auto_map(rig, game_bones):
    """{source joint name: game bone name} for the joints whose names say what they are."""
    game = {b.name: b for b in game_bones}
    out, taken = {}, {}
    for name in rig.names:
        if SKIP.search(name) and _role(name) != 'Wrist':
            continue
        role = _role(name)
        if role is None:
            continue
        side = _side(name)
        if role in SIDED:
            if side is None:
                continue
            target = '%s_%s' % (role, side)
        else:
            target = role
        if target not in game or game[target].smooth < 0:
            continue
        # two joints for one role: the one nearer the root of the source skeleton keeps it
        out.setdefault(name, target)
    return out


def resolve(rig, game_bones, bone_map, W=None):
    """Game bone index for every source joint: own mapping, else the nearest mapped ancestor's, else the nearest game
    bone by position (needs the game's world matrices W)."""
    gi = {b.name: b.index for b in game_bones}
    own = [gi.get(bone_map.get(n)) if bone_map.get(n) else None for n in rig.names]
    res = list(own)
    for i in range(len(rig.names)):
        j = i
        while res[i] is None and rig.parents[j] >= 0:
            j = rig.parents[j]
            res[i] = own[j]
    if W is not None:
        core = [b.index for b in game_bones if b.smooth >= 0 and b.name in set(bone_map.values())]
        for i, r in enumerate(res):
            if r is None and core:
                p = rig.position(i)
                res[i] = min(core, key=lambda k: np.linalg.norm(W[k][:3, 3] - p))
    return res


LIMB_FREE = ('Clavicle', 'Arm_', 'Wrist')      # bones whose direction may differ a lot between skeletons
MAX_TURN = np.radians(20.0)                     # the most the other bones are turned to match the game's direction


def _limit_turn(R, max_angle):
    """R turned by at most max_angle about its own axis."""
    c = (np.trace(R) - 1.0) / 2.0
    ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
    if ang <= max_angle or ang < 1e-9:
        return R
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return R
    x, y, z = axis / n
    s, co = np.sin(max_angle), np.cos(max_angle)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * K + (1 - co) * (K @ K)


def _axis_rot(axis, angle):
    """Rotation by `angle` radians about `axis` (right-hand rule)."""
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rot_between(a, b):
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.999999:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def explicit(rig, game_bones, bone_map):
    """Game bone index per source joint for the joints that are mapped by name (None for the rest)."""
    gi = {b.name: b.index for b in game_bones}
    return [gi.get(bone_map.get(n)) if bone_map.get(n) else None for n in rig.names]


# how closely each group of bones is pulled onto the game's joints (profile "joint_fit" overrides, by name prefix).
# Arms must meet the game's shoulders and hands; the torso, legs, neck and head keep the model's own proportions, because
# forcing a short-legged, long-torsoed model onto Link's joints squashes one and stretches the other.
DEFAULT_FIT = {'Clavicle': 0.2, 'Arm_': 0.3, 'Wrist': 0.2, 'default': 0.2}


def strength_of(name, strengths=None):
    table = dict(DEFAULT_FIT)
    table.update(strengths or {})
    for prefix, v in table.items():
        if prefix != 'default' and name.startswith(prefix):
            return float(v)
    return float(table['default'])


def bone_transforms(rig, f, W, res, src_to_world, own=None, stretch=0.18, strengths=None, arm_drop=12.0, hand_curl=15.0):
    """Per game bone (that some source joint maps to): the 4x4 that takes points of the (fitted) source rest pose into
    the game bind pose. src_to_world: 4x4 fit of the source (scale/rotation/offset) applied before."""
    n_src = len(rig.names)
    sp = np.array([(src_to_world @ np.append(rig.position(i), 1))[:3] for i in range(n_src)])
    kids_src = [[] for _ in range(n_src)]
    for i, p in enumerate(rig.parents):
        if p >= 0:
            kids_src[p].append(i)
    by_game = {}
    for i, g in enumerate(res):
        if g is not None:
            by_game.setdefault(g, []).append(i)
    mapped = set(by_game)
    kids_game = {}
    for b in f.bones:
        if b.parent >= 0:
            kids_game.setdefault(b.parent, []).append(b.index)
    depth = []
    for i in range(n_src):
        d, p = 0, rig.parents[i]
        while p >= 0:
            d, p = d + 1, rig.parents[p]
        depth.append(d)
    gname = {b.name: b.index for b in f.bones}
    T, prim = {}, {}
    for g, srcs in by_game.items():
        # the source joint standing for this game bone: the topmost of those that map to it
        named = [i for i in srcs if own is not None and own[i] == g] or srcs      # an explicitly mapped joint anchors the bone
        primary = min(named, key=lambda i: (depth[i], i))
        prim[g] = primary
        q = W[g][:3, 3]
        # direction and length of the bone on both sides, towards the mapped children (else away from the parent)
        axis = AXIS.get(f.bones[g].name)
        if axis is not None:
            gk = [gname[axis]] if axis in gname and gname[axis] in mapped else []
        elif f.bones[g].name in NO_AXIS:
            gk = []
        else:
            gk = [k for k in kids_game.get(g, []) if k in mapped][:1]
        if gk:
            gend = np.mean([W[k][:3, 3] for k in gk], axis=0)
            send = np.mean([sp[min([i for i in by_game[k] if own is not None and own[i] == k] or by_game[k], key=lambda i: (depth[i], i))] for k in gk], axis=0)
        else:
            par = f.bones[g].parent
            gdir = q - W[par][:3, 3] if par >= 0 else np.array([0, 1.0, 0])
            sk = [k for k in kids_src[primary] if res[k] != g]
            if sk:
                send = np.mean([sp[k] for k in sk], axis=0)
                gend = q + gdir / max(np.linalg.norm(gdir), 1e-9) * max(np.linalg.norm(send - sp[primary]), 0.05)
            else:
                spar = rig.parents[primary]
                sdir = sp[primary] - sp[spar] if spar >= 0 else np.array([0, 1.0, 0])
                L = max(np.linalg.norm(sdir), 0.05)
                send = sp[primary] + sdir / max(np.linalg.norm(sdir), 1e-9) * L
                gend = q + gdir / max(np.linalg.norm(gdir), 1e-9) * L
        ds, dg = send - sp[primary], gend - q
        ls, lg = max(np.linalg.norm(ds), 1e-6), max(np.linalg.norm(dg), 1e-6)
        if ls < MIN_BONE or lg < MIN_BONE or f.bones[g].name in NO_AXIS:
            R, k, u = np.eye(3), 1.0, ds / ls              # too short to say anything about direction or length
        else:
            R = _rot_between(ds, dg)
            if not f.bones[g].name.startswith(LIMB_FREE):
                R = _limit_turn(R, MAX_TURN)             # only arms may turn freely (T-pose against A-pose)
            k = float(np.clip(lg / ls, 1.0 - stretch, 1.0 + stretch))
            k = 1.0 + (k - 1.0) * strength_of(f.bones[g].name, strengths)     # no stretch where the shape is kept
            u = ds / ls
        # thickness follows the uniform fit already in sp; along the bone the length matches the game bone
        S = np.eye(3) + (k - 1.0) * np.outer(u, u)
        A = R @ S
        M = np.eye(4)
        M[:3, :3] = A
        # anchor in the middle of the bone: what the two skeletons disagree about is split between both ends of the bone
        # (a bone that is only moved, not turned or stretched, is anchored at its start)
        mid = 0.0 if (ls < MIN_BONE or lg < MIN_BONE or f.bones[g].name in NO_AXIS) else 0.5
        # how far the bone's anchor goes to the game's joint: 1 = exactly there, 0 = stays where the model has it
        st = strength_of(f.bones[g].name, strengths)
        anchor_s = sp[primary] + mid * ds
        anchor_g = (q + mid * dg) * st + anchor_s * (1.0 - st)
        M[:3, 3] = anchor_g - A @ anchor_s
        T[g] = M
    # an arm moves as ONE piece: shoulder, upper arm, forearm and hand share a transform, so the arm keeps its own shape
    # and its joints cannot be pulled apart (separate transforms for each piece kinked the elbow and wrist)
    for side in ('L', 'R'):
        chain = [gname.get(n + side) for n in ('Clavicle_', 'Arm_1_', 'Arm_2_', 'Wrist_')]
        if any(c is None or c not in T for c in chain):
            continue
        s0, s1 = sp[prim[chain[0]]], sp[prim[chain[3]]]
        g0, g1 = W[chain[0]][:3, 3], W[chain[3]][:3, 3]
        ds, dg = s1 - s0, g1 - g0
        if np.linalg.norm(ds) < MIN_BONE or np.linalg.norm(dg) < MIN_BONE:
            continue
        A = _rot_between(ds, dg)                                  # the arm is turned to the game's arm direction
        down = np.array([0.0, -1.0, 0.0])
        axis = np.cross(dg / np.linalg.norm(dg), down)            # lowers the arm towards the body
        st = strength_of('Arm_', strengths)                       # how far the shoulder is pulled onto the game's
        anchor = g0 * st + s0 * (1.0 - st)
        M = np.eye(4)
        M[:3, :3] = A
        M[:3, 3] = anchor - A @ s0
        if np.linalg.norm(axis) > 1e-3 and arm_drop:
            # the arm hangs lower: it turns about its OWN shoulder joint (the clavicle joint is nearly on the body's centre
            # line, and turning about it would swing the whole arm sideways into the body)
            pivot = (M @ np.append(sp[prim[chain[1]]], 1.0))[:3]
            Rd = _axis_rot(axis, np.radians(arm_drop))
            D = np.eye(4)
            D[:3, :3] = Rd
            D[:3, 3] = pivot - Rd @ pivot
            M = D @ M
        for c in chain:
            T[c] = M
        if hand_curl and np.linalg.norm(axis) > 1e-3:
            # the hand bends a little at the wrist, in the same direction, instead of being a straight line with the arm
            p = (M @ np.append(s1, 1.0))[:3]
            Rc = _axis_rot(axis, np.radians(hand_curl))
            H = np.eye(4)
            H[:3, :3] = Rc
            H[:3, 3] = p - Rc @ p
            T[chain[3]] = H @ M
    return T, sp


def deform(mesh, rig, res, T, src_to_world, f):
    """Move a rigged mesh into the game's bind pose and return (positions, normals, joints (n,k) matrix indices, weights)."""
    P = np.c_[mesh.pos.astype(np.float64), np.ones(len(mesh.pos))] @ src_to_world.T
    P = P[:, :3]
    Pn = np.c_[P, np.ones(len(P))]
    N = None if mesh.normal is None else mesh.normal.astype(np.float64) @ src_to_world[:3, :3].T
    J, Wt = np.asarray(mesh.joints), np.asarray(mesh.weights, np.float64)
    n, k = J.shape
    out = np.zeros((n, 3)); nout = np.zeros((n, 3)) if N is not None else None
    acc = [dict() for _ in range(n)]
    wsum = np.zeros(n)
    gb = np.array([res[j] if res[j] is not None else -1 for j in range(len(res))])
    for c in range(k):
        wj = Wt[:, c]
        for j in np.unique(J[:, c]):
            sel = (J[:, c] == j) & (wj > 0)
            g = gb[j]
            if not sel.any() or g < 0 or g not in T:
                continue
            out[sel] += wj[sel, None] * (Pn[sel] @ T[g].T)[:, :3]
            if N is not None:
                nout[sel] += wj[sel, None] * (N[sel] @ T[g][:3, :3].T)
            wsum[sel] += wj[sel]
            for r in np.nonzero(sel)[0]:
                acc[r][g] = acc[r].get(g, 0.0) + wj[r]
    lost = wsum < 1e-6
    out[lost] = P[lost]
    if N is not None:
        nout[lost] = N[lost]
        nout /= np.maximum(np.linalg.norm(nout, axis=1, keepdims=True), 1e-9)
    out[~lost] /= wsum[~lost, None]
    # weights on game bones (matrix indices), strongest first
    K = 8
    joints = np.zeros((n, K), np.int64); weights = np.zeros((n, K), np.float32)
    for r in range(n):
        items = sorted(acc[r].items(), key=lambda kv: -kv[1])[:K]
        tot = sum(w for _g, w in items) or 1.0
        for c, (g, w) in enumerate(items):
            joints[r, c] = f.bones[g].smooth
            weights[r, c] = w / tot
        if not items:
            joints[r, 0] = f.bones[next(b.index for b in f.bones if b.smooth >= 0)].smooth
            weights[r, 0] = 1.0
    return out, nout, joints, weights
