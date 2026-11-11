"""glTF / GLB reader: meshes with UVs, skin weights and the skeleton's rest pose, and the base-colour image of each material.

A skinned mesh is stored in its bind pose, so its positions are already the rest pose; the joints' rest transforms are the
inverses of the inverse bind matrices. Only the first skin is used.
"""
import json
import os
import struct
from dataclasses import dataclass

import numpy as np

from importers import Mesh

_CT = {5120: ('i1', 1), 5121: ('u1', 1), 5122: ('i2', 2), 5123: ('u2', 2), 5125: ('u4', 4), 5126: ('f4', 4)}
_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


@dataclass
class Rig:
    """A source skeleton: joint names, parent joint (-1 for roots) and the rest-pose world matrix of each joint."""
    names: list
    parents: list
    rest: np.ndarray            # (n, 4, 4)

    def position(self, i):
        return self.rest[i][:3, 3]


def _read_glb(path):
    b = open(path, 'rb').read()
    if b[:4] != b'glTF':
        return json.loads(b.decode('utf-8')), None, os.path.dirname(os.path.abspath(path))
    pos, js, binc = 12, None, None
    while pos < len(b):
        n, t = struct.unpack_from('<II', b, pos)
        chunk = b[pos + 8:pos + 8 + n]
        if t == 0x4E4F534A:
            js = json.loads(chunk.decode('utf-8'))
        elif t == 0x004E4942:
            binc = chunk
        pos += 8 + n
    return js, binc, os.path.dirname(os.path.abspath(path))


class _Doc:
    def __init__(self, path):
        self.j, self.glb_bin, self.base = _read_glb(path)
        self.buffers = {}

    def buffer(self, i):
        if i not in self.buffers:
            uri = self.j['buffers'][i].get('uri')
            self.buffers[i] = self.glb_bin if uri is None else open(os.path.join(self.base, uri), 'rb').read()
        return self.buffers[i]

    def view_bytes(self, vi):
        v = self.j['bufferViews'][vi]
        b = self.buffer(v['buffer'])
        o = v.get('byteOffset', 0)
        return b[o:o + v['byteLength']]

    def accessor(self, ai):
        a = self.j['accessors'][ai]
        dt, size = _CT[a['componentType']]
        n = _NC[a['type']]
        count = a['count']
        if 'bufferView' in a:
            v = self.j['bufferViews'][a['bufferView']]
            buf = self.buffer(v['buffer'])
            off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
            stride = v.get('byteStride') or size * n
            raw = np.frombuffer(buf, np.uint8, count * stride if stride > size * n else count * size * n, off)
            if stride > size * n:
                arr = np.lib.stride_tricks.as_strided(raw, (count, size * n), (stride, 1)).copy().view('<' + dt).reshape(count, n)
            else:
                arr = raw.view('<' + dt).reshape(count, n)
        else:
            arr = np.zeros((count, n), '<' + dt)
        if a.get('normalized'):
            arr = arr.astype(np.float32) / float(np.iinfo('<' + dt).max)
        return arr


def _node_matrix(nd):
    if 'matrix' in nd:
        return np.array(nd['matrix'], np.float64).reshape(4, 4).T
    t = np.array(nd.get('translation', [0, 0, 0]), np.float64)
    q = np.array(nd.get('rotation', [0, 0, 0, 1]), np.float64)
    s = np.array(nd.get('scale', [1, 1, 1]), np.float64)
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * s
    M[:3, 3] = t
    return M


def _world_matrices(j):
    nodes = j['nodes']
    parent = {}
    for i, nd in enumerate(nodes):
        for c in nd.get('children', []):
            parent[c] = i
    cache = {}

    def world(i):
        if i not in cache:
            m = _node_matrix(nodes[i])
            cache[i] = world(parent[i]) @ m if i in parent else m
        return cache[i]
    return world, parent


def load_gltf(path, cache_dir=None):
    """-> ([Mesh], Rig or None). Textures embedded in the file are written to cache_dir (default: next to the file)."""
    d = _Doc(path)
    j = d.j
    rig = None
    joints_nodes = []
    if j.get('skins'):
        skin = j['skins'][0]
        joints_nodes = skin['joints']
        world, parent = _world_matrices(j)
        if 'inverseBindMatrices' in skin:
            ibm = d.accessor(skin['inverseBindMatrices']).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)
            rest = np.linalg.inv(ibm)
        else:
            rest = np.array([world(n) for n in joints_nodes])
        idx = {n: k for k, n in enumerate(joints_nodes)}
        parents = []
        for n in joints_nodes:
            p = parent.get(n, -1)
            while p != -1 and p not in idx:
                p = parent.get(p, -1)
            parents.append(idx.get(p, -1))
        rig = Rig([j['nodes'][n].get('name', 'joint%d' % n) for n in joints_nodes], parents, rest)
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(path)), '_textures')
    tex_cache = {}

    def material_texture(mi):
        if mi is None:
            return None
        m = j['materials'][mi]
        t = m.get('pbrMetallicRoughness', {}).get('baseColorTexture')
        if t is None:
            return None
        img = j['images'][j['textures'][t['index']]['source']]
        if img.get('uri') and not img['uri'].startswith('data:'):
            return os.path.join(d.base, img['uri'])
        key = img.get('bufferView', id(img))
        if key not in tex_cache:
            os.makedirs(cache_dir, exist_ok=True)
            ext = '.png' if 'png' in img.get('mimeType', 'png') else '.jpg'
            p = os.path.join(cache_dir, (img.get('name') or 'image%s' % key) + ext)
            open(p, 'wb').write(d.view_bytes(img['bufferView']))
            tex_cache[key] = p
        return tex_cache[key]

    meshes = []
    for ni, nd in enumerate(j['nodes']):
        if 'mesh' not in nd:
            continue
        mesh = j['meshes'][nd['mesh']]
        skinned = 'skin' in nd and rig is not None
        M = None if skinned else _world_matrices(j)[0](ni)        # static meshes go to world space
        for pi, prim in enumerate(mesh['primitives']):
            if prim.get('mode', 4) != 4:
                continue
            at = prim['attributes']
            pos = d.accessor(at['POSITION']).astype(np.float64)
            nrm = d.accessor(at['NORMAL']).astype(np.float64) if 'NORMAL' in at else None
            if M is not None:
                pos = pos @ M[:3, :3].T + M[:3, 3]
                if nrm is not None:
                    nrm = nrm @ np.linalg.inv(M[:3, :3])
            uv = d.accessor(at['TEXCOORD_0']).astype(np.float32) if 'TEXCOORD_0' in at else None
            if uv is not None:
                uv = np.c_[uv[:, 0], 1.0 - uv[:, 1]]                    # glTF has v down; Mesh.uv is v up as in OBJ
            tris = (d.accessor(prim['indices']).reshape(-1).astype(np.int64).reshape(-1, 3) if 'indices' in prim
                    else np.arange(len(pos)).reshape(-1, 3))
            jn = wt = None
            if skinned and 'JOINTS_0' in at:
                jn = [d.accessor(at['JOINTS_0']).astype(np.int64)]
                wt = [d.accessor(at['WEIGHTS_0']).astype(np.float32)]
                if 'JOINTS_1' in at:
                    jn.append(d.accessor(at['JOINTS_1']).astype(np.int64))
                    wt.append(d.accessor(at['WEIGHTS_1']).astype(np.float32))
                jn, wt = np.concatenate(jn, 1), np.concatenate(wt, 1)
            mat = j['materials'][prim['material']] if 'material' in prim else {}
            meshes.append(Mesh('%s.%d' % (nd.get('name', mesh.get('name', 'mesh')), pi), mat.get('name', ''),
                               pos.astype(np.float32), tris, uv, None if nrm is None else nrm.astype(np.float32),
                               material_texture(prim.get('material')), jn, wt))
    return meshes, rig
