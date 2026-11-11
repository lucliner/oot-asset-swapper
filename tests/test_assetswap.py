"""tools/assetswap's readers on made-up data (no game files): the attribute format table, the 10_10_10_2 decoder, vertex
attribute decoding and the SARC reader. The reader itself was checked against the real models by hand (see
tools/assetswap); this keeps the pieces from drifting."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'assetswap'))
import bfres  # noqa: E402
import autoskin  # noqa: E402
import bfres_write  # noqa: E402
import importers  # noqa: E402
from containers import sarc_read  # noqa: E402

RUN_ALL_MODULES = ["numpy", "zstandard"]


def check(cond, what):
    if not cond:
        raise SystemExit('FAIL: ' + what)


def test_formats():
    # the formats the game's models use, with the component width and count they were measured to have
    want = {0x1805: ('f4', 3, 'float'), 0x0e02: ('p1010102', 4, 'snorm'), 0x0b01: ('u1', 4, 'unorm'),
            0x1205: ('f2', 2, 'float'), 0x1201: ('u2', 2, 'unorm'), 0x1503: ('u2', 4, 'uint'),
            0x0903: ('u1', 2, 'uint'), 0x0a03: ('u2', 1, 'uint'), 0x0201: ('u1', 1, 'unorm'),
            0x1505: ('f2', 4, 'float'), 0x0b02: ('i1', 4, 'snorm')}
    for fmt, w in want.items():
        check(bfres.fmt_info(fmt) == w, 'fmt_info(0x%04x) = %r, want %r' % (fmt, bfres.fmt_info(fmt), w))
    check(bfres.fmt_info(0x7f01) is None, 'unknown type must give None')


def make(attrs, rows, stride):
    """A Bfres shell holding one shape's vertex buffer, for read_attr."""
    f = bfres.Bfres.__new__(bfres.Bfres)
    f.b = b''.join(rows)
    s = bfres.Shape(0, 't', 0, 0, 0, 0, len(rows), attrs, stride, 0, 0, 0, 0, 0)
    return f, s


def test_decode():
    def pack(x, y, z, w):
        q = lambda v: int(round(v * 511)) & 0x3ff
        return struct.pack('<I', q(x) | q(y) << 10 | q(z) << 20 | (w & 3) << 30)
    rows = [struct.pack('<3f', 1, 2, 3) + pack(0, 1, -1, 1) + struct.pack('<4B', 255, 0, 51, 255) + struct.pack('<4H', 1, 2, 3, 280),
            struct.pack('<3f', -4, 5, 6) + pack(0.5, -0.5, 0, 0) + struct.pack('<4B', 0, 0, 0, 0) + struct.pack('<4H', 0, 0, 0, 0)]
    attrs = [bfres.Attr('_p0', 0x1805, 0, 0), bfres.Attr('_n0', 0x0e02, 12, 0),
             bfres.Attr('_w0', 0x0b01, 16, 0), bfres.Attr('_i0', 0x1503, 20, 0)]
    f, s = make(attrs, rows, 28)
    p = f.read_attr(s, '_p0')
    check(np.allclose(p, [[1, 2, 3], [-4, 5, 6]]), 'position decode')
    n = f.read_attr(s, '_n0')
    check(np.allclose(n[0, :3], [0, 1, -1], atol=1e-6) and n[0, 3] == 1, 'snorm 10_10_10_2 decode: %r' % n[0])
    check(np.allclose(n[1, :3], [0.5, -0.5, 0], atol=2e-3), '10_10_10_2 second vertex: %r' % n[1])
    w = f.read_attr(s, '_w0')
    check(np.allclose(w[0], [1, 0, 0.2, 1]), 'unorm8 weights: %r' % w[0])
    i = f.read_attr(s, '_i0')
    check(i.dtype.kind == 'u' and i[0].tolist() == [1, 2, 3, 280], '16-bit bone indices (a 280 does not fit in 8 bits): %r' % i[0])
    check(f.read_attr(s, '_zz') is None, 'a missing attribute reads as None')


def test_sarc():
    names = [b'a.bin', b'b.bin']
    datas = [b'1234', b'56']
    sfnt = b'SFNT' + struct.pack('<HH', 8, 0)
    offs = []
    for n in names:
        offs.append(len(sfnt) - 8)
        sfnt += n + b'\0' * (4 - len(n) % 4)
    sfat = b'SFAT' + struct.pack('<HHI', 0x0c, 2, 0x65)
    pos = 0
    for k, (n, d) in enumerate(zip(names, datas)):
        sfat += struct.pack('<IIII', k, 0x01000000 | (offs[k] // 4), pos, pos + len(d))
        pos += len(d)
    data_off = 0x14 + len(sfat) + len(sfnt)
    hdr = b'SARC' + struct.pack('<HHIIHH', 0x14, 0xfeff, data_off + pos, data_off, 0x100, 0)
    blob = hdr + sfat + sfnt + b''.join(datas)
    check(sarc_read(blob) == {'a.bin': b'1234', 'b.bin': b'56'}, 'sarc_read: %r' % sarc_read(blob))


def test_encode_roundtrip():
    rng = np.random.default_rng(1)
    n = 50
    cases = {0x1805: rng.normal(size=(n, 3)).astype(np.float32), 0x1205: rng.uniform(-3, 3, (n, 2)),
             0x1201: rng.uniform(0, 1, (n, 2)), 0x0b01: rng.uniform(0, 1, (n, 4)),
             0x1503: rng.integers(0, 300, (n, 4)), 0x0b03: rng.integers(0, 255, (n, 4)), 0x0a03: rng.integers(0, 300, (n, 1))}
    for fmt, v in cases.items():
        data, size = bfres_write.encode_attr(fmt, v)
        f, s = make([bfres.Attr('_p0', fmt, 0, 0)], [data[k * size:(k + 1) * size] for k in range(n)], size)
        got = f.read_attr(s, '_p0')
        tol = {0x1205: 2e-3, 0x1201: 1e-4, 0x0b01: 3e-3}.get(fmt, 1e-6)
        check(np.allclose(got, np.asarray(v).reshape(n, -1), atol=tol * max(1, np.abs(v).max())), 'encode/decode 0x%04x' % fmt)
    nrm = rng.normal(size=(n, 3)); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    data, size = bfres_write.encode_attr(0x0e02, np.c_[nrm, np.ones(n)])
    f, s = make([bfres.Attr('_n0', 0x0e02, 0, 0)], [data[k * 4:(k + 1) * 4] for k in range(n)], 4)
    g = f.read_attr(s, '_n0')
    check(np.abs(g[:, :3] - nrm).max() < 3e-3 and (g[:, 3] == 1).all(), '10_10_10_2 encode/decode')
    try:
        bfres_write.encode_attr(0x0b03, np.full((1, 4), 300))
        check(False, 'a bone index over 255 must not fit an 8-bit index attribute')
    except ValueError:
        pass
    w = bfres_write._quantize_weights(np.array([[0.333, 0.333, 0.334, 0.0]]), 4)
    check(abs(w.sum() - 1) < 1e-9 and np.allclose(w * 255, np.rint(w * 255)), 'weights quantise to exact bytes summing to 255')


def test_obj_and_autoskin():
    import tempfile
    nl = chr(10)
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'm.mtl'), 'w').write(nl.join(['newmtl a', 'map_Kd t.png', '']))
        open(os.path.join(d, 'm.obj'), 'w').write(nl.join([
            'mtllib m.mtl', 'o q', 'usemtl a', 'v 0 0 0', 'v 1 0 0', 'v 1 1 0', 'v 0 1 0',
            'vt 0 0', 'vt 1 0', 'vt 1 1', 'vt 0 1', 'f 1/1 2/2 3/3 4/4', 'f -4/1 -3/2 -2/3', '']))
        ms = importers.load_obj(os.path.join(d, 'm.obj'))
        check(len(ms) == 1 and len(ms[0].tris) == 3 and ms[0].material == 'a' and ms[0].texture.endswith('t.png'), 'OBJ: quad fan, negative index, mtl')
        check(ms[0].uv is not None and len(ms[0].pos) == 4, 'OBJ: corners shared by position/uv pairs stay merged')

    # a made-up skeleton: a thigh and a shin stacked on y; points near each take that bone
    class B:
        def __init__(s, i, n, p, sm): s.index, s.name, s.parent, s.smooth = i, n, p, sm
    class F:
        bones = [B(0, 'Waist', -1, 7), B(1, 'Leg_1_L', 0, 8), B(2, 'Leg_2_L', 1, 9), B(3, 'Ankle_L', 2, 10)]
    W = [np.eye(4) for _ in range(4)]
    for i, y in enumerate([1.0, 1.0, 0.5, 0.05]):
        W[i][1, 3] = y
    pts = np.array([[0, 0.9, 0], [0, 0.55, 0], [0, 0.45, 0], [0, 0.02, 0.1]])
    tris = np.array([[0, 1, 2], [1, 2, 3]])
    j, w = autoskin.autoskin(F, W, pts, tris, smooth=0)
    check(j.shape == (4, 4) and np.allclose(w.sum(1), 1), 'autoskin weights sum to 1')
    check(j[0, 0] in (7, 8) and j[3, 0] == 10, 'autoskin: the strongest bone is the nearest: %r' % j[:, 0])
    ms2 = [importers.Mesh('m', 'a', np.array([[0, 0, 0], [0, 2, 0], [1, 1, 0]], np.float32), np.array([[0, 1, 2]]))]
    sc, R, t = autoskin.fit(ms2, F, W, target_height=1.0)
    check(abs(sc - 0.5) < 1e-9 and abs(t[1]) < 1e-9, 'fit: scale to the target height, feet on the ground')


def make_glb():
    """A one-triangle GLB with a two-joint skin (joint 1 is a child of joint 0, one metre up)."""
    import json
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], '<f4')
    uv = np.array([[0, 0], [1, 0], [0, 1]], '<f4')
    jn = np.array([[0, 1, 0, 0]] * 3, '<u1')
    wt = np.array([[0.75, 0.25, 0, 0]] * 3, '<f4')
    ib = np.array([np.eye(4), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, -1, 0, 1]])], '<f4')   # column-major
    idx = np.array([0, 1, 2], '<u2')
    parts = [pos.tobytes(), uv.tobytes(), jn.tobytes(), wt.tobytes(), ib.tobytes(), idx.tobytes() + b'\0\0']
    views, off = [], 0
    for p in parts:
        views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(p)})
        off += len(p)
    acc = [{'bufferView': 0, 'componentType': 5126, 'count': 3, 'type': 'VEC3'},
           {'bufferView': 1, 'componentType': 5126, 'count': 3, 'type': 'VEC2'},
           {'bufferView': 2, 'componentType': 5121, 'count': 3, 'type': 'VEC4'},
           {'bufferView': 3, 'componentType': 5126, 'count': 3, 'type': 'VEC4'},
           {'bufferView': 4, 'componentType': 5126, 'count': 2, 'type': 'MAT4'},
           {'bufferView': 5, 'componentType': 5123, 'count': 3, 'type': 'SCALAR'}]
    doc = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': off}], 'bufferViews': views, 'accessors': acc,
           'nodes': [{'name': 'Hips', 'children': [1]}, {'name': 'Spine', 'translation': [0, 1, 0]}, {'name': 'M', 'mesh': 0, 'skin': 0}],
           'skins': [{'joints': [0, 1], 'inverseBindMatrices': 4}],
           'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'TEXCOORD_0': 1, 'JOINTS_0': 2, 'WEIGHTS_0': 3}, 'indices': 5}]}]}
    js = json.dumps(doc).encode()
    js += b' ' * (-len(js) % 4)
    binc = b''.join(parts)
    binc += b'\0' * (-len(binc) % 4)
    total = 12 + 8 + len(js) + 8 + len(binc)
    return (b'glTF' + struct.pack('<II', 2, total) + struct.pack('<II', len(js), 0x4E4F534A) + js
            + struct.pack('<II', len(binc), 0x004E4942) + binc)


def test_gltf_and_retarget():
    import tempfile
    import gltf_io
    import retarget
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'm.glb')
        open(p, 'wb').write(make_glb())
        meshes, rig = gltf_io.load_gltf(p, d)
    check(len(meshes) == 1 and len(meshes[0].pos) == 3 and len(meshes[0].tris) == 1, 'glb: one triangle')
    check(rig.names == ['Hips', 'Spine'] and rig.parents == [-1, 0], 'glb: joints and parents: %r' % (rig.parents,))
    check(np.allclose(rig.position(1), [0, 1, 0]), 'glb: rest position of a joint comes from the inverse bind matrix')
    check(np.allclose(meshes[0].weights[0][:2], [0.75, 0.25]) and meshes[0].joints.shape == (3, 4), 'glb: skin weights')
    check(meshes[0].uv is not None and abs(meshes[0].uv[2, 1] - 0.0) < 1e-6, 'glb: v is flipped to the OBJ convention')

    # name matching: MMD, Mixamo and Blender-style names land on the game's bone names
    class B:
        def __init__(s, n, sm=1): s.name, s.smooth = n, sm
    game = [B(n) for n in ('Waist', 'Spine_1', 'Spine_2', 'Neck', 'Head', 'Clavicle_L', 'Arm_1_L', 'Arm_2_L', 'Wrist_L',
                           'Leg_1_R', 'Leg_2_R', 'Ankle_R', 'Toe_R')]
    class Rg:
        names = ['mixamorig:Hips', 'UpperBody', 'UpperBody2', 'neck_01', 'head', 'Shoulder_L', 'Arm_L', 'Elbow_L', 'HandTwist_L',
                 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LegIK_R', 'IndexFinger1_L', 'Lower_Arm.L']
    m = retarget.auto_map(Rg, game)
    for k, v in {'UpperBody': 'Spine_1', 'UpperBody2': 'Spine_2', 'neck_01': 'Neck', 'head': 'Head', 'Shoulder_L': 'Clavicle_L',
                 'Arm_L': 'Arm_1_L', 'Elbow_L': 'Arm_2_L', 'HandTwist_L': 'Wrist_L', 'Lower_Arm.L': 'Arm_2_L'}.items():
        check(m.get(k) == v, 'auto_map %s -> %s, got %s' % (k, v, m.get(k)))
    check('LegIK_R' not in m and 'IndexFinger1_L' not in m, 'auto_map leaves IK and finger joints out')


def test_restbl():
    import restbl
    pairs = sorted([(restbl.Restbl.key('Tex/x.bntx'), 1000), (5, 7)])       # the table is sorted by hash
    tbl = b'RESTBL' + struct.pack('<IIII', 1, 0xa0, 2, 0) + b''.join(struct.pack('<II', h, s) for h, s in pairs)
    t = restbl.Restbl(tbl)
    check(t.to_bytes() == tbl, 'RESTBL round trip')
    check(t.get('Tex/x.bntx.zs') == 1000, 'RESTBL lookup ignores the .zs suffix')
    t.set('Tex/x.bntx.zs', 1234)
    check(restbl.Restbl(t.to_bytes()).get('Tex/x.bntx') == 1234, 'RESTBL set')


def test_block_encoders():
    """BC1 and ASTC encoders checked by an independent decoder (texture2ddecoder, a dev-only package: skipped without it)."""
    import block_encode
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:96, 0:96]
    img = np.stack([xx * 2.5, yy * 2.5, (xx + yy) * 1.2], -1).astype(np.uint8)
    img[30:60, 30:60] = (230, 40, 40)                                   # a hard-edged patch
    check(block_encode.encode_bc1(img).shape == (24, 24, 8), 'BC1 block array shape')
    check(block_encode.encode_astc(img, 8, 8).shape == (12, 12, 16), 'ASTC block array shape')
    for bw, bh in ((8, 8), (10, 8), (12, 12)):
        wg, hg, _l, _a, _b, qb = block_encode.best_config(bw, bh)
        check(24 <= wg * hg * qb <= 63 and wg <= bw and hg <= bh, 'ASTC %dx%d: a legal weight grid within 63 bits' % (bw, bh))
    try:
        import texture2ddecoder as t2d
    except ImportError:
        print('(texture2ddecoder not installed: the decoder cross-check is skipped)')
        return

    def psnr(a, b):
        mse = ((a.astype(float) - b.astype(float)) ** 2).mean()
        return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    dec = np.frombuffer(t2d.decode_bc1(block_encode.encode_bc1(img).tobytes(), 96, 96), np.uint8).reshape(96, 96, 4)[..., [2, 1, 0]]
    check(psnr(img, dec) > 30, 'BC1 round trip through an independent decoder: %.1f dB' % psnr(img, dec))
    dec = np.frombuffer(t2d.decode_astc(block_encode.encode_astc(img, 8, 8).tobytes(), 96, 96, 8, 8), np.uint8).reshape(96, 96, 4)[..., [2, 1, 0]]
    check(psnr(img, dec) > 25, 'ASTC 8x8 round trip through an independent decoder: %.1f dB' % psnr(img, dec))


def test_backup():
    import tempfile
    import time
    import backup
    with tempfile.TemporaryDirectory() as md:
        mod = os.path.join(md, 'm')
        os.makedirs(os.path.join(mod, 'romfs', 'Model'))
        open(os.path.join(mod, 'romfs', 'Model', 'a.bin'), 'wb').write(b'version one')
        check(backup.snapshot(md, 'nothing') is None, 'a mod that does not exist has nothing to back up')
        s1 = backup.snapshot(md, 'm', 'first')
        check(s1 and os.path.isfile(os.path.join(s1, 'romfs', 'Model', 'a.bin')), 'snapshot copies the mod')
        check(backup.DIRNAME.startswith('_'), 'the backups folder name is one the game\'s mod loader skips')
        open(os.path.join(mod, 'romfs', 'Model', 'a.bin'), 'wb').write(b'version two')
        time.sleep(1.1)
        backup.restore(md, s1)
        check(open(os.path.join(mod, 'romfs', 'Model', 'a.bin'), 'rb').read() == b'version one', 'restore brings the old files back')
        snaps = backup.list_snapshots(md, 'm')
        check(len(snaps) == 2 and any('before restoring' in s[3].get('reason', '') for s in snaps),
              'a restore backs up what it replaces first: %r' % [s[3].get('reason') for s in snaps])
        off = backup.set_enabled(md, 'm', False)
        check(os.path.basename(off) == '-m' and backup.mods(md) == [('m', False)], 'switching off renames to -m')
        backup.set_enabled(md, 'm', True)
        snap = backup.remove(md, 'm')
        check(snap and not os.path.exists(mod) and backup.mods(md) == [], 'remove deletes the mod and keeps a backup')
        dest = backup.restore(md, snap)
        check(os.path.isfile(os.path.join(dest, 'romfs', 'Model', 'a.bin')), 'a removed mod can be restored')
        for i in range(backup.KEEP + 3):
            backup.snapshot(md, 'm', 'n%d' % i)
        check(len(backup.list_snapshots(md, 'm')) <= backup.KEEP, 'old backups are pruned')


if __name__ == '__main__':
    for t in (test_formats, test_decode, test_sarc, test_encode_roundtrip, test_obj_and_autoskin, test_gltf_and_retarget, test_restbl,
              test_block_encoders, test_backup):
        t()
    print('assetswap reader checks passed')
