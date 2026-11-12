"""OpenGL rendering of the previews (moderngl, off-screen): smooth-shaded, depth-tested, antialiased, and fast enough to turn a
model with the mouse. The camera is the same orthographic one preview.Camera describes, so everything that projects points with
it (joint handles, the skeleton overlay) lines up with what is drawn here.

A context belongs to the thread that made it, so each thread gets its own renderer (`get()`). When OpenGL is not available
`get()` returns None and callers fall back to the CPU painter in preview.py.
"""
import threading

import numpy as np
from PIL import Image

import preview

try:
    import moderngl
except Exception:                                            # not installed: the CPU painter is used instead
    moderngl = None

_VS = """
#version 330
uniform mat3 R;
uniform mat4 xf;       // the model's own move / turn / scale; the identity for the original model
uniform vec4 cam;      // centre x, lowest y, scale, depth middle
uniform vec3 view;     // width, height, depth scale
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col; in vec2 in_uv;
out vec3 v_col; out float v_nz; out vec2 v_uv;
void main() {
    vec3 Q = R * (xf * vec4(in_pos, 1.0)).xyz;
    vec3 N = normalize(R * (mat3(xf) * in_nrm));
    float px = (Q.x - cam.x) * cam.z + view.x * 0.5;
    float py = view.y - (Q.y - cam.y) * cam.z - view.x * 0.06;
    gl_Position = vec4(2.0 * px / view.x - 1.0, 1.0 - 2.0 * py / view.y, -(Q.z - cam.w) * view.z, 1.0);
    v_col = in_col; v_nz = abs(N.z); v_uv = in_uv;
}
"""
_FS = """
#version 330
uniform float alpha;
uniform vec3 tint;     // multiplies the colour; (1, 1, 1) leaves it
uniform sampler2D tex;
uniform int use_tex;
in vec3 v_col; in float v_nz; in vec2 v_uv;
out vec4 f;
void main() {
    float k = 0.32 + 0.68 * v_nz;
    vec3 c = use_tex == 1 ? texture(tex, v_uv).rgb : v_col;
    f = vec4(c * tint * k, alpha);
}
"""

_local = threading.local()


def get():
    """This thread's renderer, or None when OpenGL cannot be used."""
    if moderngl is None or getattr(_local, 'failed', False):
        return None
    r = getattr(_local, 'r', None)
    if r is None:
        try:
            r = _local.r = Renderer()
        except Exception:
            _local.failed = True
            return None
    return r


def smooth_normals(tris):
    """Per-vertex normals for (n, 3, 3) triangles: the area-weighted mean over the triangles that meet at a point."""
    n = len(tris)
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    pts = tris.reshape(-1, 3)
    key = np.round(pts * 2000.0).astype(np.int64)
    _u, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((inv.max() + 1, 3))
    np.add.at(acc, inv, np.repeat(fn, 3, axis=0))
    out = acc[inv]
    ln = np.linalg.norm(out, axis=1, keepdims=True)
    flat = np.repeat(fn, 3, axis=0)
    out = np.where(ln > 1e-12, out / np.maximum(ln, 1e-12), flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12))
    return out.astype(np.float32)


class Renderer:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self._buf = {}                                       # id of a triangle array -> (vao, vbo, count)
        self._tex = {}
        self._fbo = None

    def _vao(self, tris, colors, smooth, uv=None, nrm_in=None):
        key = (id(tris), tris.shape, id(colors), smooth, id(uv), id(nrm_in))
        hit = self._buf.get(key)
        if hit is not None and hit[3] is tris:
            return hit
        if len(self._buf) > 90:
            for v in self._buf.values():
                v[1].release(); v[0].release()
            self._buf.clear()
        t = np.asarray(tris, np.float32)
        nrm = np.asarray(nrm_in, np.float32).reshape(-1, 3) if nrm_in is not None else smooth_normals(t.astype(np.float64)) if smooth else np.repeat(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), 3, axis=0).astype(np.float32)
        if not smooth and nrm_in is None:
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        if colors is None:
            col = np.empty((len(t) * 3, 3), np.float32); col[:] = (0.93, 0.93, 0.84)
        else:
            col = np.repeat(np.asarray(colors, np.float32) / 255.0, 3, axis=0)
        uvs = np.zeros((len(t) * 3, 2), np.float32) if uv is None else np.asarray(uv, np.float32).reshape(-1, 2)
        data = np.concatenate([t.reshape(-1, 3), nrm.reshape(-1, 3), col, uvs], axis=1).astype('f4')
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 3f 3f 2f', 'in_pos', 'in_nrm', 'in_col', 'in_uv')])
        self._buf[key] = hit = (vao, vbo, len(t) * 3, tris)
        return hit

    def _texture(self, arr):
        hit = self._tex.get(id(arr))
        if hit is not None and hit[1] is arr:
            return hit[0]
        if len(self._tex) > 400:
            for t, _a in self._tex.values():
                t.release()
            self._tex.clear()
        a = np.ascontiguousarray(arr[::-1])                   # image row 0 is the top; GL row 0 is the bottom
        t = self.ctx.texture((a.shape[1], a.shape[0]), 3, a.tobytes())
        t.build_mipmaps(); t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR); t.repeat_x = t.repeat_y = True
        self._tex[id(arr)] = (t, arr)
        return t

    def _target(self, w, h):
        if self._fbo is None or self._fbo[0] != (w, h):
            if self._fbo is not None:
                for o in self._fbo[1:]:
                    o.release()
            ms = self.ctx.texture((w, h), 4, samples=4)
            md = self.ctx.depth_renderbuffer((w, h), samples=4)
            msf = self.ctx.framebuffer([ms], md)
            rt = self.ctx.texture((w, h), 4)
            rf = self.ctx.framebuffer([rt])
            self._fbo = ((w, h), ms, md, msf, rt, rf)
        return self._fbo

    def render(self, tris, yaw=0.0, pitch=0.0, size=640, ref=None, colors=None, xray=False, ghost=None, smooth=True,
               opacity=0.30, ghost_opacity=0.40, parts=None, xform=None):
        """(PIL image, preview.Camera). tris (n, 3, 3); ghost: a second triangle set drawn tinted and see-through on top.
        parts: [{'tris', 'uv', 'tex'}] instead of tris: each part is drawn with its own texture."""
        if parts:
            tris = np.concatenate([q['tris'] for q in parts if len(q['tris'])]) if any(len(q['tris']) for q in parts) else np.zeros((0, 3, 3))
        else:
            parts = [{'tris': np.asarray(tris), 'uv': None, 'tex': None, 'colors': colors}]
        tris = np.asarray(tris)
        both = tris if ghost is None or not len(ghost) else np.concatenate([tris, ghost])
        cam = preview.Camera(tris if len(tris) else np.zeros((1, 3, 3)), yaw, pitch, size, both if ref is None else ref)
        w, h = cam.w, cam.h
        ctx = self.ctx
        _s, _ms, _md, msf, _rt, rf = self._target(w, h)
        pts = (ref if ref is not None else both).reshape(-1, 3) @ cam.R.T
        zlo, zhi = float(pts[:, 2].min()), float(pts[:, 2].max())
        zmid, zs = (zlo + zhi) / 2, 1.0 / max((zhi - zlo) / 2 * 1.05, 1e-6)
        p = self.prog
        p['R'].write(np.ascontiguousarray(cam.R.T, np.float32).tobytes())          # GLSL reads column-major
        p['cam'].value = (float(cam.cx), float(cam.lo[1]), float(cam.scale), zmid)
        p['view'].value = (float(w), float(h), zs)
        msf.use()
        bg = (15 / 255, 17 / 255, 21 / 255) if xray else (24 / 255, 24 / 255, 28 / 255)
        ctx.clear(*bg, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        if xray:                                               # the model faded into the background: no depth, plain blend
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            p['alpha'].value = opacity
        else:
            p['alpha'].value = 1.0
        p['tint'].value = (1.0, 1.0, 1.0)
        eye = np.eye(4, dtype=np.float32).tobytes()
        p['xf'].write(eye if xform is None else np.ascontiguousarray(np.asarray(xform, np.float64).T, np.float32).tobytes())
        for q in parts:
            if not len(q['tris']):
                continue
            use = q.get('tex') is not None and q.get('uv') is not None
            vao, _vbo, count, _keep = self._vao(q['tris'], q.get('colors'), smooth, q.get('uv') if use else None, q.get('nrm'))
            if use:
                self._texture(q['tex']).use(0)
                p['tex'].value = 0
            p['use_tex'].value = 1 if use else 0
            vao.render(moderngl.TRIANGLES, vertices=count)
        p['use_tex'].value = 0
        p['xf'].write(eye)                                      # the original model never moves
        if ghost is not None and len(ghost):
            gv, _b, gc, _k = self._vao(ghost, None, True)
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            p['alpha'].value = ghost_opacity
            p['tint'].value = (90 / 255 * 1.9, 200 / 255 * 1.2, 1.0)
            gv.render(moderngl.TRIANGLES, vertices=gc)
        ctx.disable(moderngl.BLEND)
        ctx.copy_framebuffer(rf, msf)                          # resolve the multisampled image
        raw = rf.read(components=3)
        img = Image.frombytes('RGB', (w, h), raw).transpose(Image.FLIP_TOP_BOTTOM)
        return img, cam
