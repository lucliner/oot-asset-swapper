"""The swap itself: an imported model replaces the geometry of one of the game's character models, as a mod.

A `profile` (a plain dict, saved as JSON by the UI) holds everything the user can correct:

    scale, rotate_y, offset      fit of the model to the target (scale default: the target's height)
    bone_map                     {source joint: game bone, or null} corrections on top of the automatic name matching
    slots                        {part name: shape index} which of the target's shapes carries which part
    texture_map                  {material name: image path} base-colour images (the model's own when it has them)
    attachments                  {game bone: [x, y, z]} local positions of attachment bones (weapons), or "auto"
    hide                         game shape indices to keep drawing (default: all other shapes are emptied)
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

from containers import Romfs
from bfres import Bfres, fmt_info
from bfres_write import patch_model, SlotMesh
from skeleton import world_matrices
import autoskin
import gltf_io
import importers
import retarget

# shapes whose materials need special shading (alpha, hair, face and eye effects) make poor hosts for an opaque body
AVOID = re.compile(r'eye|lash|brow|hair|teeth|tongue|lip|face|alpha|softedge|shadow|feather', re.I)
WEAPON_BONES = {'L': 'Weapon_L', 'R': 'Weapon_R'}
EQUIPMENT_PODS = ('Pod_A', 'Pod_B')          # where the shield and the sword are mounted on Link's back


def find_blender():
    for p in (os.environ.get('BLENDER'), r'E:\Blender\blender.exe', r'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe',
              r'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe'):
        if p and os.path.exists(p):
            return p
    return None


def load_model(path, cache_dir=None, log=print):
    """Read a model of any format: OBJ and glTF directly, anything else through Blender (converted to GLB first).
    -> ([Mesh], Rig or None)."""
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'assetswap_cache')
    ext = os.path.splitext(path)[1].lower()
    if ext == '.obj':
        return importers.load_obj(path), None
    if ext not in ('.glb', '.gltf'):
        blender = find_blender()
        if not blender:
            raise RuntimeError('%s needs Blender to convert it (set the BLENDER environment variable to blender.exe)' % ext)
        os.makedirs(cache_dir, exist_ok=True)
        out = os.path.join(cache_dir, os.path.splitext(os.path.basename(path))[0] + '.glb')
        if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
            log('converting %s with Blender...' % os.path.basename(path))
            here = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))      # (the packaged app keeps it beside the code)
            r = subprocess.run([blender, '-b', '-P', os.path.join(here, 'blender_convert.py'),
                                '--', path, out], capture_output=True, text=True)
            if not os.path.exists(out):
                raise RuntimeError('Blender could not convert the model:\n' + r.stdout[-800:] + r.stderr[-400:])
        path = out
    return gltf_io.load_gltf(path, os.path.join(cache_dir, 'textures'))


def pick_slots(f, count, max_matrix, need_float_uv=False, rom=None):
    """Shapes of the target that can carry an imported mesh, best first: plain opaque materials with a full vertex layout,
    the most skin influences, bone indices wide enough for the bones used, and float UVs when the mesh tiles. With `rom`,
    only shapes whose base-colour texture is the model's own (a shared cloth tile cannot be recoloured)."""
    import textures as tex
    ranked = []
    for s in f.shapes:
        if not s.vertex_count or s.skin_count < 4 or AVOID.search(s.name + ' ' + f.materials[s.material]):
            continue
        if rom is not None and not tex.own_albedo(rom, f, s.material):
            continue
        names = {a.name: a for a in s.attrs}
        if not all(n in names for n in ('_p0', '_n0', '_t0', '_u0', '_i0', '_w0')):
            continue
        if 8 * int(fmt_info(names['_i0'].fmt)[0][1]) == 8 and max_matrix > 255:
            continue
        cap = sum(fmt_info(a.fmt)[1] for a in s.attrs if a.name in ('_i0', '_i1'))
        ranked.append((need_float_uv and fmt_info(names['_u0'].fmt)[2] != 'float', -cap, -s.vertex_count, s.index))
    ranked.sort()
    return [r[3] for r in ranked[:count]]


class Plan:
    """Everything a swap needs, computed from a profile. `slots` maps shape index -> SlotMesh (bind pose, game bones)."""

    def __init__(self, romfs_dir, target_rel, input_path, profile=None, log=print):
        self.log = log
        self.profile = dict(profile or {})
        self.rom = Romfs(romfs_dir)
        self.target_rel = target_rel
        self.src = self.rom.read(target_rel)
        self.f = Bfres(self.src)
        self.W = world_matrices(self.f)
        self.meshes, self.rig = load_model(input_path, log=log)
        p = self.profile
        self.top = max(float(self.f.read_attr(s, '_p0')[:, 1].max()) for s in self.f.shapes if s.vertex_count and s.skin_count > 1)
        if self.rig is not None:
            self.auto_bone_map = retarget.auto_map(self.rig, self.f.bones)
            self.bone_map = dict(self.auto_bone_map)
            self.bone_map.update(p.get('bone_map') or {})
            self.bone_map = {k: v for k, v in self.bone_map.items() if v}
        else:
            self.auto_bone_map, self.bone_map = {}, {}
        self.slots, self.bone_positions, self.part_slot = {}, {}, {}
        self._build()

    # -- fit
    def _fit(self):
        p = self.profile
        ry = np.radians(p.get('rotate_y') or 0.0)
        R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        P = np.concatenate([m.pos for m in self.meshes]).astype(np.float64) @ R.T
        lo, hi = P.min(0), P.max(0)
        sc = p.get('scale') or self.top / max(hi[1] - lo[1], 1e-6)
        t = np.array([-(lo[0] + hi[0]) / 2 * sc, -lo[1] * sc, 0.0]) + np.array(p.get('offset') or (0, 0, 0))
        M = np.eye(4)
        M[:3, :3] = R * sc
        M[:3, 3] = t
        return sc, R, t, M

    def _merge_parts(self, parts, skip):
        """Parts named in the profile's merge_parts (e.g. ["damage"]) are joined to the biggest textured part instead of
        taking a shape of their own: each vertex is given the texel of the host's atlas whose colour is closest to the
        colour the part's own texture has there. For small flat-coloured pieces (cuffs, hems, caps that close the gaps of
        the main mesh) that is all the texture they need."""
        names = [n.lower() for n in (self.profile.get('merge_parts') or [])]
        if not names:
            return parts
        from PIL import Image, ImageDraw
        tmap = self.profile.get('texture_map') or {}
        img_of = lambda m: tmap.get(m.material) or m.texture
        merge = [m for m in parts if any(n in (m.name + ' ' + m.material).lower() for n in names) and m.uv is not None and img_of(m)]
        if not merge:
            return parts
        rest = [m for m in parts if m not in merge]
        hosts = [m for m in rest if m.uv is not None and img_of(m) and m.joints is not None and not any(x in (m.name + ' ' + m.material).lower() for x in skip)]
        if not hosts:
            return parts
        into = str(self.profile.get('merge_into') or '').lower()
        if into:
            hosts = [m for m in hosts if into in (m.name + ' ' + m.material).lower()] or hosts
        host = max(hosts, key=lambda m: len(m.pos))
        himg = np.asarray(Image.open(img_of(host)).convert('RGB'))
        # (a part laid out in another tile of the texture is moved into 0..1 first, as the slot code does)
        lo = np.floor((1.0 - host.uv[:, 1]).min() + 1e-3)
        if lo:
            host.uv = host.uv.copy()
            host.uv[:, 1] += lo
        H, W_ = himg.shape[:2]
        # the host's used texels, away from island edges (so a flat piece never samples a neighbour's colour)
        mask = Image.new('L', (W_, H), 0)
        dr = ImageDraw.Draw(mask)
        huv = np.c_[host.uv[:, 0], 1.0 - host.uv[:, 1]] * [W_, H]
        for t in np.asarray(host.tris):
            dr.polygon([tuple(q) for q in huv[t]], fill=255)
        used = np.asarray(mask) > 0
        er = used.copy()
        for _ in range(3):
            e = er.copy()
            e[1:] &= er[:-1]; e[:-1] &= er[1:]; e[:, 1:] &= er[:, :-1]; e[:, :-1] &= er[:, 1:]
            er = e
        ys, xs = np.where(er if er.sum() > 1000 else used)
        pick = np.random.RandomState(1).permutation(len(ys))[:60000]
        ys, xs = ys[pick], xs[pick]
        cols = himg[ys, xs].astype(np.float64)
        for m in merge:
            dimg = np.asarray(Image.open(img_of(m)).convert('RGB'))
            dh, dw = dimg.shape[:2]
            px = np.clip((m.uv[:, 0] * dw).astype(int), 0, dw - 1)
            py = np.clip(((1.0 - m.uv[:, 1]) * dh).astype(int), 0, dh - 1)
            want = dimg[py, px].astype(np.float64)
            new_uv = np.zeros((len(m.pos), 2), np.float32)
            for c in np.unique(want, axis=0):
                sel = (want == c).all(1)
                k = np.argmin(((cols - c) ** 2).sum(1))
                new_uv[sel] = [(xs[k] + 0.5) / W_, 1.0 - (ys[k] + 0.5) / H]
            k1, k2 = host.joints.shape[1], m.joints.shape[1]
            kk = max(k1, k2)
            pad = lambda a, k: np.pad(a, ((0, 0), (0, k - a.shape[1])))
            host.joints = np.concatenate([pad(host.joints, kk), pad(m.joints, kk)])
            host.weights = np.concatenate([pad(host.weights, kk), pad(m.weights, kk)])
            host.tris = np.concatenate([host.tris, m.tris + len(host.pos)])
            host.pos = np.concatenate([host.pos, m.pos])
            host.uv = np.concatenate([host.uv, new_uv])
            if host.normal is not None and m.normal is not None:
                host.normal = np.concatenate([host.normal, m.normal])
            else:
                host.normal = None
            self.meshes = [x for x in self.meshes if x is not m]
            self.log('  %-30s merged into %s (%d verts, colours matched to its atlas)' % (m.name[:30], host.name[:24], len(m.pos)))
        return rest

    def _build(self):
        p, f, W, log = self.profile, self.f, self.W, self.log
        sc, R, t, M = self._fit()
        self.M, self.scale = M, sc
        log('fit: scale %.4f, rotate_y %.0f deg' % (sc, p.get('rotate_y') or 0))
        max_matrix = max(b.smooth for b in autoskin.core_bones(f))
        need_float = any(m.uv is not None and (m.uv.min() < -1e-4 or m.uv.max() > 1 + 1e-4) for m in self.meshes)
        parts = sorted(self.meshes, key=lambda m: -len(m.pos))
        want = dict(p.get('slots') or {})
        want.update((p.get('slots_by_target') or {}).get(f.model_name, {}))      # shape numbers differ from model to model
        # parts left out: any the user set to -1, and by name ("damage": the torn-clothes overlay of a model, shown only
        # when it is hurt, would otherwise be drawn all the time)
        skip = [s.lower() for s in p.get('skip_parts', ['damage'])]
        parts = self._merge_parts(parts, skip)
        kept = []
        for m in parts:
            if want.get(m.name) == -1 or any(s in (m.name + ' ' + m.material).lower() for s in skip):
                log('  %-30s left out (%s)' % (m.name[:30], 'chosen' if want.get(m.name) == -1 else 'matches skip_parts'))
                self.skipped = getattr(self, 'skipped', []) + [m.name]
            else:
                kept.append(m)
        parts = kept
        free = (list(p['slots_list']) if p.get('slots_list')
                else [s for s in pick_slots(f, len(parts) + len(want) + 8, max_matrix, need_float, self.rom) if s not in want.values()])
        import textures as _tex
        used_albedo = set()
        albedo_of = lambda s: _tex.albedo_name(f, f.shapes[s].material)
        self.res = self.T = None
        if self.rig is not None and any(m.joints is not None for m in parts):
            self.res = retarget.resolve(self.rig, f.bones, self.bone_map, W)
            self.own = retarget.explicit(self.rig, f.bones, self.bone_map)
            drop = p.get('arm_drop', -25.0)          # ('auto' tries to measure the body; it cannot tell a sleeve from a belly in a one-piece shirt)
            curl = p.get('hand_curl', 15.0)
            if drop in (None, 'auto'):
                drop = self._auto_arm_drop([m for m in parts if m.joints is not None], M, curl)
            self.arm_drop = float(drop)
            self.T, _sp = retarget.bone_transforms(self.rig, f, W, self.res, M, self.own, strengths=p.get('joint_fit'),
                                                   arm_drop=self.arm_drop, hand_curl=curl)
            log('rigged source: %d joints, %d mapped to game bones; arm drop %.0f deg' % (len(self.rig.names), len(self.bone_map), self.arm_drop))
        for m in parts:
            si = want.get(m.name)
            if si is None:
                # a shape whose albedo file another part already uses is skipped: one file can hold one image
                pick = next((s for s in free if albedo_of(s) not in used_albedo or albedo_of(s) is None), None)
                if pick is None:
                    if not self.slots:
                        raise ValueError('the target has no usable shape slot; pick another target')
                    log('  %-30s skipped: the target has no more shapes with an albedo of their own' % m.name[:30])
                    self.skipped = getattr(self, 'skipped', []) + [m.name]
                    continue
                free.remove(pick)
                si = pick
            used_albedo.add(albedo_of(si))
            self.part_slot[m.name] = si
            uv = None if m.uv is None else np.c_[m.uv[:, 0], 1.0 - m.uv[:, 1]]
            if uv is not None:
                # a part laid out in another tile of a repeating texture (v from 1 to 2, or -1 to 0) is shifted into 0..1,
                # so it does not depend on the material repeating the texture
                lo = np.floor(uv.min(0) + 1e-3)
                if ((uv.max(0) - lo) <= 1.0 + 1e-3).all():
                    uv = uv - lo
            tex = (p.get('texture_map') or {}).get(m.material) or m.texture
            if m.joints is not None and self.T is not None:
                pos, nrm, j, w = retarget.deform(m, self.rig, self.res, self.T, M, f)
            else:
                pos = (m.pos.astype(np.float64) @ R.T) * sc + t
                nrm = None if m.normal is None else m.normal.astype(np.float64) @ R.T
                j, w = autoskin.autoskin(f, W, pos, m.tris)
            tris = np.asarray(m.tris)
            if nrm is not None:
                # a triangle wound against its own normals is culled as a back face (it shows as a hole in the cloth): turn it round
                P3 = pos.astype(np.float64)
                gn = np.cross(P3[tris[:, 1]] - P3[tris[:, 0]], P3[tris[:, 2]] - P3[tris[:, 0]])
                bad = (gn * np.asarray(nrm, np.float64)[:, :3][tris].sum(1)).sum(1) < 0
                if bad.any():
                    tris = tris.copy()
                    tris[bad] = tris[bad][:, ::-1]
                    log('  %d triangles wound against their normals were turned round' % bad.sum())
            self.slots[si] = SlotMesh(pos.astype(np.float32), tris, j, w, nrm, uv, texture=tex, uv_float=need_float)
            log('  %-30s -> shape %2d %-36s (%d verts, %d tris)' % (m.name[:30], si, f.shapes[si].name[:36], len(pos), len(m.tris)))
        self._flatten_back()
        self._stiffen_arms()
        self._fit_skeleton()
        self._attachments()

    def _auto_arm_drop(self, parts, M, curl, ceiling=25.0, floor=-25.0, step=3.0, tolerance=0.012):
        """The most the arms can hang down (degrees) without going through the body: the torso's outline (how wide it is at
        each height) is measured, then the arms are tried from `ceiling` downwards until their lower part stays outside it."""
        f, W = self.f, self.W
        gname = {b.name: b.index for b in f.bones}
        chain = {gname[n + s] for n in ('Clavicle_', 'Arm_1_', 'Arm_2_', 'Wrist_') for s in 'LR' if n + s in gname}
        if len(chain) < 8:
            return 12.0
        strengths = self.profile.get('joint_fit')

        def transforms(d):
            return retarget.bone_transforms(self.rig, f, W, self.res, M, self.own, strengths=strengths, arm_drop=d, hand_curl=curl)[0]
        T0 = transforms(0.0)
        gb = np.array([self.res[j] if self.res[j] is not None else -1 for j in range(len(self.res))])
        arm_parts, torso_pts = [], []
        trunk = [gname[n] for n in ('Spine_1', 'Spine_2', 'Waist', 'Neck') if n in gname]
        # (the clavicle bones also hold the upper chest in many rigs, so only the arm proper is judged)
        limbs = {gname[n + s] for n in ('Arm_1_', 'Arm_2_', 'Wrist_') for s in 'LR' if n + s in gname}
        for m in parts:
            frac = np.zeros(len(m.pos)); tfrac = np.zeros(len(m.pos))
            for c in range(m.joints.shape[1]):
                frac += np.where(np.isin(gb[m.joints[:, c]], list(limbs)), m.weights[:, c], 0.0)
                tfrac += np.where(np.isin(gb[m.joints[:, c]], trunk), m.weights[:, c], 0.0)
            arm = frac > 0.5
            pos0 = retarget.deform(m, self.rig, self.res, T0, M, f)[0]
            torso_pts.append(pos0[tfrac > 0.6])                  # the trunk only: sleeves are not part of its outline
            if arm.any():
                arm_parts.append(importers.Mesh('arm', '', m.pos[arm], np.zeros((0, 3), np.int64), None, None, None,
                                                m.joints[arm], m.weights[arm]))
        if not arm_parts:
            return 12.0
        torso = np.concatenate(torso_pts)
        shoulders = [W[gname[n]][:3, 3] for n in ('Arm_1_L', 'Arm_1_R') if n in gname]          # the arms' own shoulder joints
        shoulder_y = max(s[1] for s in shoulders)
        torso = torso[(torso[:, 1] < shoulder_y) & (torso[:, 1] > 0.2)]
        bins = np.floor(torso[:, 1] / 0.02).astype(int)
        half = {}
        for b_, x in zip(bins, np.abs(torso[:, 0])):
            half[b_] = max(half.get(b_, 0.0), x)

        def penetration(d):
            T = transforms(d)
            worst = []
            for am in arm_parts:
                pos = retarget.deform(am, self.rig, self.res, T, M, f)[0]
                # only the outer arm counts: next to the shoulder the arm is meant to be joined to the body
                dist = np.min([np.linalg.norm(pos - sh, axis=1) for sh in shoulders], axis=0)
                low = pos[(pos[:, 1] < shoulder_y - 0.04) & (dist > 0.10)]
                for x, y in zip(np.abs(low[:, 0]), low[:, 1]):
                    w_ = half.get(int(np.floor(y / 0.02)))
                    if w_ is not None and x < w_:
                        worst.append(w_ - x)
            return float(np.percentile(worst, 95)) if worst else 0.0
        best = (None, 1e9)
        d = ceiling
        while d >= floor - 1e-9:
            pen = penetration(d)
            if pen < tolerance:
                self.log('  arms: %.0f deg of drop keeps them outside the body' % d)
                return d
            if pen < best[1]:
                best = (d, pen)
            d -= step
        self.log('  arms: no angle clears the body completely; using %.0f deg (%.1f cm in)' % (best[0], best[1] * 100))
        return best[0]

    # -- weapon / attachment bones follow the model's own extremities, or the profile's positions
    def _attachments(self):
        self._weapon_points()
        self._center_equipment()
        self._clear_equipment()
        self._offsets()

    def _flatten_back(self):
        """The game puts the shield and sword on Link's back at a fixed place: moving the mount bones has no effect in
        game (their positions come from the animation), so a body deeper than Link's runs into them. The back of the
        torso is therefore pressed in until it is no deeper than Link's own back at that height (profile flatten_back:
        false keeps the model's back; flatten_margin is how many metres behind Link's back it may stay)."""
        p = self.profile
        if p.get('flatten_back', True) is False or not self.slots:
            return
        f = self.f
        margin = float(p.get('flatten_margin', 0.0))
        # Link's own back: the deepest point behind the spine, per height
        L = np.concatenate([np.asarray(f.read_attr(s, '_p0'), np.float64)[:, :3] for s in f.shapes if s.vertex_count and s.skin_count > 1])
        L = L[np.abs(L[:, 0]) < 0.2]
        edges = np.arange(0.25, 1.01, 0.05)
        prof = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (L[:, 1] >= a) & (L[:, 1] < b) & (L[:, 2] < 0.05)
            prof.append(L[sel, 2].min() if sel.any() else 0.0)
        centres = (edges[:-1] + edges[1:]) / 2
        prof = np.convolve(np.pad(prof, 1, mode='edge'), np.ones(3) / 3, 'valid')
        torso = {b.smooth for b in f.bones if b.smooth >= 0 and (b.name.startswith('Spine') or b.name.startswith('Waist') or b.name.startswith('Skl_Root'))}
        pressed = 0
        for sl in self.slots.values():
            wts = np.asarray(sl.weights)
            dom = np.argmax(wts, axis=1)
            jn = np.asarray(sl.joints)[np.arange(len(dom)), dom]
            P = np.asarray(sl.pos, np.float64).copy()
            limit = np.interp(P[:, 1], centres, prof) - margin
            sel = np.isin(jn, list(torso)) & (P[:, 2] < limit) & (np.abs(P[:, 0]) < 0.3) & (P[:, 1] > 0.25)
            if sel.any():
                P[sel, 2] = limit[sel] + (P[sel, 2] - limit[sel]) * 0.1      # what lies behind the limit is squeezed to a sliver
                sl.pos = P.astype(np.float32)
                pressed += int(sel.sum())
        if pressed:
            self.log('  back of the torso pressed in at %d vertices (to the depth of Links own back)' % pressed)

    def _stiffen_arms(self):
        """Part of the weight of the forearm and hand moves to the upper arm (profile arm_stiffness, 0..1, default 0.6):
        the game's animations bend the elbow and wrist by their own, Link-sized amounts, which on a thicker, shorter arm
        folds it into a zigzag. 0 leaves the weights as the model has them, 1 makes each arm turn as one piece."""
        s_ = float(self.profile.get('arm_stiffness', 0.6))
        if s_ <= 0 or not self.slots:
            return
        names = {b.name: b for b in self.f.bones}
        pair = {}
        for side in ('L', 'R'):
            a1 = names.get('Arm_1_' + side)
            if a1 is None or a1.smooth < 0:
                continue
            for n in ('Arm_2_' + side, 'Wrist_' + side):
                b = names.get(n)
                if b is not None and b.smooth >= 0:
                    pair[b.smooth] = a1.smooth
        if not pair:
            return
        moved = 0
        for sl in self.slots.values():
            J = np.asarray(sl.joints).copy()
            Wt = np.asarray(sl.weights, np.float64).copy()
            rows = np.where(np.isin(J, list(pair)).any(1))[0]
            for r in rows:
                add = {}
                for c in range(J.shape[1]):
                    j = int(J[r, c])
                    if j in pair and Wt[r, c] > 0:
                        t = Wt[r, c] * s_
                        Wt[r, c] -= t
                        add[pair[j]] = add.get(pair[j], 0.0) + t
                for j, t in add.items():
                    hit = np.where((J[r] == j) & (Wt[r] > 0))[0]
                    if len(hit):
                        Wt[r, hit[0]] += t
                    else:
                        free = np.where(Wt[r] <= 0)[0]
                        if not len(free):
                            free = [int(np.argmin(Wt[r]))]
                        J[r, free[0]] = j
                        Wt[r, free[0]] = t
                tot = Wt[r].sum()
                if tot > 0:
                    Wt[r] /= tot
            moved += len(rows)
            sl.joints = J.astype(np.asarray(sl.joints).dtype)
            sl.weights = Wt.astype(np.float32)
        self.log('  arms stiffened (%.0f%%) at %d vertices' % (s_ * 100, moved))

    def _fit_skeleton(self):
        """The game's skeleton takes the source model's proportions: each mapped bone is moved to where its source joint
        ended up in the fitted mesh (profile fit_skeleton: true, experimental: Link's animations also key bone positions, which distorts the body in game; off by default). Animations turn the
        bones about those positions, so the shoulders, elbows and hips of the new body bend where its own joints are;
        the inverse bind matrices of every bone that moved are rewritten to match."""
        p = self.profile
        self.inv_bind = {}
        fit = p.get('fit_skeleton', False)
        if not fit or self.rig is None or self.T is None or not self.slots:
            return
        only = tuple(fit) if isinstance(fit, (list, tuple)) else None      # a list of bone name prefixes limits the fit to those bones
        f = self.f
        target = {}
        for i, g in enumerate(self.own):
            if g is not None and g in self.T and g not in target and (only is None or self.f.bones[g].name.startswith(only)):
                J = self.M @ np.append(self.rig.position(i), 1.0)
                target[g] = (self.T[g] @ J)[:3]
        if not target:
            return
        from skeleton import euler_matrix
        Wn = []
        moved = 0
        for b in f.bones:
            L = np.eye(4)
            L[:3, :3] = euler_matrix(*b.rot[:3]) * np.array(b.scale)
            L[:3, 3] = b.pos
            par = Wn[b.parent] if b.parent >= 0 else np.eye(4)
            if b.index in target:
                local = np.linalg.inv(par) @ np.append(target[b.index], 1.0)
                L[:3, 3] = local[:3]
                self.bone_positions[b.index] = tuple(float(x) for x in local[:3])
                moved += 1
            Wn.append(par @ L)
        for b in f.bones:
            if b.smooth >= 0 and not np.allclose(Wn[b.index], self.W[b.index], atol=1e-7):
                self.inv_bind[b.smooth] = np.linalg.inv(Wn[b.index])[:3, :]
        self.W = Wn
        self.log('  skeleton fitted to the model: %d bones moved' % moved)

    def _bone_world(self, bone):
        """Where an attachment bone is now: its planned local position (or the game's) carried by its parent."""
        b = self.f.bones[bone]
        local = self.bone_positions.get(bone, tuple(b.pos))
        return (self.W[b.parent] @ np.append(local, 1.0))[:3] if b.parent >= 0 else np.array(local)

    def _set_world(self, bone, world):
        b = self.f.bones[bone]
        local = np.linalg.inv(self.W[b.parent]) @ np.append(world, 1.0) if b.parent >= 0 else np.append(world, 1.0)
        self.bone_positions[bone] = tuple(float(x) for x in local[:3])

    def _center_equipment(self):
        """The shield and sword mounts follow the model's own torso, not Link's: the pods are moved sideways and up or down
        so the shield disc is centred on the part of the model the spine bones carry (profile equipment_center: false
        keeps Link's positions; attach_offsets still adds on top)."""
        p = self.profile
        if p.get('equipment_center', True) is False or not self.slots:
            return
        names = {b.name: b for b in self.f.bones}
        spine = [names[n].smooth for n in ('Spine_1', 'Spine_2') if n in names and names[n].smooth >= 0]
        pa = names.get('Pod_A')
        if not spine or pa is None:
            return
        pts = []
        for sl in self.slots.values():
            wts = np.asarray(sl.weights)
            dom = np.argmax(wts, axis=1)
            jn = np.asarray(sl.joints)[np.arange(len(dom)), dom]
            sel = np.isin(jn, spine) & (wts[np.arange(len(dom)), dom] > 0.5)
            if sel.any():
                pts.append(np.asarray(sl.pos, np.float64)[sel])
        if not pts:
            return
        pts = np.concatenate(pts)
        lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        centre = (lo + hi) / 2
        a = self._bone_world(pa.index)
        shield_dy = 0.07                                   # the shield's centre sits 7 cm above its mount (measured from its model)
        delta = np.array([centre[0] - a[0], centre[1] - shield_dy - a[1], 0.0])
        for name in EQUIPMENT_PODS:
            b = names.get(name)
            if b is not None:
                self._set_world(b.index, self._bone_world(b.index) + delta)
        self.log('  equipment mounts centred on the torso (moved %.0f cm sideways, %.0f cm up)' % (delta[0] * 100, delta[1] * 100))

    def _clear_equipment(self):
        """Back-mounted equipment (the shield and sword sit on Pod_A and Pod_B) is pushed out behind the model until its
        mount point is `equipment_clearance` metres off the model's surface there: a body deeper than Link's would
        otherwise swallow half of it."""
        p = self.profile
        if p.get('equipment_auto_clear', True) is False or not self.slots:
            return
        clearance = float(p.get('equipment_clearance', 0.28))      # the shield reaches 0.13 m toward the body from its mount
        P = np.concatenate([np.asarray(s.pos, np.float64) for s in self.slots.values()])
        names = {b.name: b for b in self.f.bones}
        for name in EQUIPMENT_PODS:
            b = names.get(name)
            if b is None:
                continue
            w = self._bone_world(b.index)
            r = float(p.get('equipment_radius', 0.30))          # the Deku shield is a disc 0.5 m across (measured from its model)
            near = P[(np.abs(P[:, 1] - w[1]) < r) & (np.abs(P[:, 0] - w[0]) < r)]
            if not len(near):
                continue
            target = near[:, 2].min() - clearance                       # behind the back at that height
            if w[2] > target:                                            # only ever pushed out, never pulled in
                w = w.copy(); w[2] = target
                self._set_world(b.index, w)
                self.log('  %s moved %.0f cm out of the model' % (name, (self.W[b.index][2, 3] - target) * 100))

    def _offsets(self):
        """Manual offsets (metres, model space) added to attachment bones: profile attach_offsets {bone: [x, y, z]}."""
        names = {b.name: b for b in self.f.bones}
        for name, d in (self.profile.get('attach_offsets') or {}).items():
            b = names.get(name)
            if b is not None and any(abs(float(x)) > 1e-9 for x in d):
                self._set_world(b.index, self._bone_world(b.index) + np.array([float(x) for x in d]))
                self.log('  %s offset by %s' % (name, [round(float(x), 3) for x in d]))

    def _weapon_points(self):
        f, W = self.f, self.W
        spec = self.profile.get('attachments', 'auto')
        names = {b.name: b for b in f.bones}
        if isinstance(spec, dict):
            for n, pos in spec.items():
                if n in names:
                    self.bone_positions[names[n].index] = tuple(pos)
            return
        if spec != 'auto':
            return
        for side, wname in WEAPON_BONES.items():
            wrist, elbow, weapon = names.get('Wrist_' + side), names.get('Arm_2_' + side), names.get(wname)
            if not (wrist and elbow and weapon):
                continue
            pts = []
            for sl in self.slots.values():
                dom = np.argmax(np.asarray(sl.weights), axis=1)
                jn = np.asarray(sl.joints)[np.arange(len(dom)), dom]
                sel = (jn == wrist.smooth) & (np.asarray(sl.weights)[np.arange(len(dom)), dom] > 0.5)
                if sel.any():
                    pts.append(np.asarray(sl.pos, np.float64)[sel])
            if not pts:
                continue
            pts = np.concatenate(pts)
            e = W[elbow.index][:3, 3]
            far = np.linalg.norm(pts - e, axis=1)
            tip = pts[np.argsort(far)[-max(1, len(pts) // 25):]].mean(0)       # the outermost few percent
            point = e + (tip - e) * 0.9                                           # a little inside the end
            local = np.linalg.inv(W[weapon.parent]) @ np.append(point, 1.0)
            self.bone_positions[weapon.index] = tuple(float(x) for x in local[:3])
            self.log('  %s moved to the end of the %s limb' % (wname, side))

    def overlays(self):
        """Lines/points for the preview: the game's body bones (cyan) and, for a rigged source, each mapped joint's fitted
        position (orange) with a line to the game bone it is mapped to, so a wrong mapping shows as a long line."""
        f, W = self.f, self.W
        ov = []
        core = autoskin.core_bones(f)
        lines = [(W[b.parent][:3, 3], W[b.index][:3, 3]) for b in f.bones if b.parent >= 0 and b in core and f.bones[b.parent] in core]
        ov.append(('lines', np.array(lines) if lines else [], (80, 200, 255)))
        if self.rig is not None:
            gi = {b.name: b.index for b in f.bones}
            pts, links = [], []
            for i, n in enumerate(self.rig.names):
                g = self.bone_map.get(n)
                if g in gi:
                    p = (self.M @ np.append(self.rig.position(i), 1.0))[:3]
                    pts.append(p); links.append((p, W[gi[g]][:3, 3]))
            ov.append(('lines', np.array(links) if links else [], (255, 150, 60)))
            ov.append(('points', np.array(pts) if pts else [], (255, 190, 90)))
        return ov

    # -- output
    def write(self, mod_dir, textures=True, log=None):
        """Write the mod: the model, the models to hide, and (when asked) the textures. Returns the written model's Bfres."""
        log = log or self.log
        rom, f = self.rom, self.f
        import backup
        snap = backup.snapshot(os.path.dirname(mod_dir), os.path.basename(mod_dir), 'before a build')
        if snap:
            log('backed up the existing mod first: %s' % snap)
        packed = rom.compress(self.model_bytes())
        dest = os.path.join(mod_dir, 'romfs', *self.target_rel.split('/'))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as fh:
            fh.write(packed)
        back = Bfres(rom.decompress(open(dest, 'rb').read()))
        assert len(back.shapes) == len(f.shapes) and [b.name for b in back.bones] == [b.name for b in f.bones]
        log('wrote %s (%d bytes; the original is %d)' % (dest, len(packed), os.path.getsize(rom.path(self.target_rel))))
        for rel in self.profile.get('hide_models') or []:
            try:
                data = patch_model(rom.read(rel), {})
            except Exception as e:                                   # a model this writer cannot read is left alone
                log('  could not empty %s: %s' % (rel, e))
                continue
            p = os.path.join(mod_dir, 'romfs', *rel.split('/'))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'wb').write(rom.compress(data, dict_name=rom.dict_of(rel)))
            log('  hidden: %s' % rel)
        if textures:
            import textures as tex
            files = tex.build(rom, f, {si: s.texture for si, s in self.slots.items()}, log,
                              {si: (s.uv, s.tris) for si, s in self.slots.items()})
            for rel, data in files.items():
                p = os.path.join(mod_dir, 'romfs', *rel.split('/'))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, 'wb') as fh:
                    fh.write(data)
            # (the files keep the donors' sizes, so the game's resource size table needs no change)
        return back

    def model_bytes(self):
        return patch_model(self.src, self.slots, bone_positions=self.bone_positions, inv_bind=getattr(self, 'inv_bind', None))

    def triangle_colors(self, images=None):
        """Per-triangle colour for every slot in `triangles` order: the slot's base-colour image sampled at the triangle's
        UV centre (a rough look at how the texture lines up). images: {shape index: image path} overrides."""
        from PIL import Image
        out = []
        for si, sl in self.slots.items():
            n = len(sl.tris)
            path = (images or {}).get(si) or sl.texture
            if not path or sl.uv is None or not os.path.exists(path):
                out.append(np.full((n, 3), 170, np.uint8))
                continue
            im = np.asarray(Image.open(path).convert('RGBA'), np.uint8)
            h, w = im.shape[:2]
            uv = np.asarray(sl.uv, np.float64)[np.asarray(sl.tris)].mean(1)
            x = np.clip((uv[:, 0] % 1.0) * w, 0, w - 1).astype(int)
            y = np.clip((uv[:, 1] % 1.0) * h, 0, h - 1).astype(int)
            out.append(im[y, x, :3])
        return np.concatenate(out) if out else np.zeros((0, 3), np.uint8)

    def triangles(self, pose=False):
        """The planned model's triangles (n, 3, 3), bind pose or the test pose (linear blend skinning)."""
        from posing import pose_matrices, rot_about  # noqa: F401
        f = self.f
        M = pose_matrices(f, self.W) if pose else None
        out = []
        for sl in self.slots.values():
            P = np.asarray(sl.pos, np.float64)
            if pose:
                h = np.c_[P, np.ones(len(P))]
                Q = np.zeros_like(P)
                J, Wt = np.asarray(sl.joints), np.asarray(sl.weights, np.float64)
                for c in range(J.shape[1]):
                    for m_ in np.unique(J[:, c]):
                        sel = (J[:, c] == m_) & (Wt[:, c] > 0)
                        if sel.any():
                            Q[sel] += Wt[sel, c, None] * (h[sel] @ M[f.matrix_to_bone[m_]].T)[:, :3]
                P = Q
            out.append(P[np.asarray(sl.tris)])
        return np.concatenate(out) if out else np.zeros((0, 3, 3))


def swap(romfs_dir, target_rel, input_path, mod_dir, scale=None, rotate_y=0.0, offset=(0, 0, 0), slots=None, textures=True,
         log=print, profile=None):
    """Write mod_dir/romfs/<target_rel> with the imported model in place of the target's geometry. Returns the Plan."""
    prof = dict(profile or {})
    for k, v in (('scale', scale), ('rotate_y', rotate_y), ('offset', offset)):
        if v not in (None, 0.0, (0, 0, 0)):
            prof.setdefault(k, v)
    if slots:
        prof.setdefault('slots_list', slots)
    plan = Plan(romfs_dir, target_rel, input_path, prof, log)
    plan.back = plan.write(mod_dir, textures, log)
    return plan
