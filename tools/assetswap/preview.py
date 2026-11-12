"""Software preview of models (flat shaded, painter's algorithm), with optional skeleton overlays."""
import numpy as np
from PIL import Image, ImageDraw


class Camera:
    def __init__(self, tris, yaw=0.0, pitch=0.0, size=640, ref=None):
        """ref: points used to frame the view (defaults to the triangles), so overlays do not change the framing."""
        self.R = self._rot(yaw, pitch)
        pts = (tris if ref is None else ref).reshape(-1, 3) @ self.R.T
        self.lo, self.hi = pts.min(0), pts.max(0)
        span = max(self.hi[1] - self.lo[1], self.hi[0] - self.lo[0], 1e-6)
        self.scale = size / span * 0.88
        self.w = size
        self.h = int(size * max(self.hi[1] - self.lo[1], 1e-6) / span * 0.88 + size * 0.12)
        self.cx = (self.lo[0] + self.hi[0]) / 2

    @staticmethod
    def _rot(yaw, pitch):
        y, p = np.radians(yaw), np.radians(pitch)
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rp = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        return Rp @ Ry

    def project(self, P):
        """(n, 3) world points -> (n, 2) pixels and (n,) depth."""
        Q = np.asarray(P, np.float64) @ self.R.T
        x = (Q[:, 0] - self.cx) * self.scale + self.w / 2
        y = self.h - (Q[:, 1] - self.lo[1]) * self.scale - self.w * 0.06
        return np.c_[x, y], Q[:, 2]


def render_image(tris, yaw=0.0, pitch=0.0, size=640, overlays=(), ref=None, colors=None):
    """tris: (n, 3, 3). colors: optional (n, 3) uint8 per-triangle base colour (texture sampled at the triangle's centre).
    overlays: [(kind, data, colour)] with kind 'lines' (m, 2, 3), 'points' (m, 3)."""
    cam = Camera(tris if len(tris) else np.zeros((1, 3, 3)), yaw, pitch, size, ref)
    img = Image.new('RGB', (cam.w, cam.h), (24, 24, 28))
    d = ImageDraw.Draw(img)
    if len(tris):
        t = tris @ cam.R.T
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        shade = 0.35 + 0.65 * np.abs(n[:, 2])
        order = np.argsort(t[:, :, 2].mean(1))
        xy, _ = cam.project(tris.reshape(-1, 3))
        xy = xy.reshape(-1, 3, 2)
        for i in order:
            if colors is not None:
                k = 0.45 + 0.55 * shade[i]
                d.polygon([tuple(p) for p in xy[i]], fill=tuple(int(x * k) for x in colors[i]))
            else:
                c = int(60 + 190 * shade[i])
                d.polygon([tuple(p) for p in xy[i]], fill=(c, c, int(c * 0.9)))
    for kind, data, colour in overlays:
        if kind == 'lines' and len(data):
            a, _ = cam.project(np.asarray(data)[:, 0]); b, _ = cam.project(np.asarray(data)[:, 1])
            for p, q in zip(a, b):
                d.line([tuple(p), tuple(q)], fill=colour, width=2)
        elif kind == 'points' and len(data):
            p, _ = cam.project(np.asarray(data))
            for x, y in p:
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colour)
    return img


# ---- x-ray: a see-through model with the skeleton drawn solid on top, and joint handles that can be dragged --------------

def render_base(tris, yaw=0.0, pitch=0.0, size=640, ref=None, colors=None, xray=False, opacity=0.30, ghost=None, parts=None, xform=None):
    """The model alone (no skeleton), as (image, camera). With xray the model is blended into the background at `opacity`
    so what is drawn over it afterwards stays solid and readable. ghost: the game's own model as a see-through overlay.
    Drawn with OpenGL when it is available, else by the CPU painter below."""
    try:
        import glview
        gl = glview.get()
        if gl is not None and len(tris):
            return gl.render(tris, yaw, pitch, size, ref=ref, colors=colors, xray=xray, ghost=ghost, opacity=opacity, parts=parts, xform=xform)
    except Exception:
        pass
    img = render_image(tris, yaw, pitch, size, (), ref, colors)
    cam = Camera(tris if len(tris) else np.zeros((1, 3, 3)), yaw, pitch, size, ref)
    if xray:
        bg = Image.new('RGB', img.size, (15, 17, 21))
        a = np.asarray(img)
        covered = (a != np.array([24, 24, 28], np.uint8)).any(-1)                      # (the empty background is (24, 24, 28))
        mask = Image.fromarray((covered * int(255 * opacity)).astype(np.uint8))
        img = Image.composite(img, bg, mask)
    if ghost is not None:
        img = add_ghost(img, ghost, yaw, pitch, size, ref)
    return img, cam


def draw_skeleton(img, cam, overlays=(), handles=(), active=None):
    """Draw bones and handles on a copy of img. handles: [(key, (x, y, z) world point, (r, g, b))]. Returns (image, {key: (px, py)})."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    for kind, data, colour in overlays:
        if kind == 'lines' and len(data):
            a, _ = cam.project(np.asarray(data)[:, 0]); b, _ = cam.project(np.asarray(data)[:, 1])
            for p, q in zip(a, b):
                d.line([tuple(p), tuple(q)], fill=colour, width=4)
                d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=colour)
    pos = {}
    for key, point, colour in handles:
        xy, _z = cam.project(np.asarray([point], np.float64))
        x, y = float(xy[0][0]), float(xy[0][1])
        pos[key] = (x, y)
        r = 9 if key == active else 7
        d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255) if key == active else colour)
    return out, pos


def add_ghost(img, ghost_tris, yaw, pitch, size, ref, opacity=0.40, tint=(90, 200, 255)):
    """The game's own model laid over img as a tinted see-through silhouette (same camera: ref must be what img was framed on)."""
    if not len(ghost_tris):
        return img
    g = render_image(ghost_tris, yaw, pitch, size, (), ref)
    a = np.asarray(g)
    covered = (a != np.array([24, 24, 28], np.uint8)).any(-1)
    shade = a.astype(np.float64).mean(-1, keepdims=True) / 255.0
    color = Image.fromarray(np.clip(np.asarray(tint, np.float64) * (0.55 + 0.6 * shade), 0, 255).astype(np.uint8))
    mask = Image.fromarray((covered * int(255 * opacity)).astype(np.uint8))
    out = img.copy()
    out.paste(color, (0, 0), mask)
    return out


AXES = {'x': (np.array([1.0, 0, 0]), (235, 80, 80)), 'y': (np.array([0, 1.0, 0]), (90, 205, 100)), 'z': (np.array([0, 0, 1.0]), (90, 140, 255))}


def draw_gizmo(img, cam, center, radius, mode, active=None):
    """The transform handles on a copy of img. mode: 'move' (an arrow per axis and a centre square), 'rotate' (a ring per axis),
    'scale' (a cube per axis and the centre). Returns (image, geometry): the pixel points of what can be grabbed."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    c = np.asarray(center, np.float64)
    cpx = cam.project(c[None])[0][0]
    geo = {'center': (float(cpx[0]), float(cpx[1])), 'rings': {}, 'handles': {}}
    if mode == 'rotate':
        t = np.linspace(0, 2 * np.pi, 72)
        for k, (a, col) in AXES.items():
            u, v = [e for kk, (e, _c) in AXES.items() if kk != k]
            pts = c + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
            px = cam.project(pts)[0]
            geo['rings'][k] = px
            w = 6 if k == active else 4
            d.line([tuple(q) for q in px], fill=(255, 255, 255) if k == active else col, width=w)
    else:
        for k, (a, col) in AXES.items():
            end = c + a * radius
            e = cam.project(end[None])[0][0]
            geo['handles'][k] = (float(e[0]), float(e[1]))
            colr = (255, 255, 255) if k == active else col
            d.line([tuple(cpx), tuple(e)], fill=colr, width=4)
            r = 8 if mode == 'scale' else 7
            if mode == 'scale':
                d.rectangle([e[0] - r, e[1] - r, e[0] + r, e[1] + r], fill=colr, outline=(20, 20, 24))
            else:
                d.ellipse([e[0] - r, e[1] - r, e[0] + r, e[1] + r], fill=colr, outline=(20, 20, 24))
        r = 9
        colr = (255, 255, 255) if active == 'c' else (240, 240, 240)
        d.rectangle([cpx[0] - r, cpx[1] - r, cpx[0] + r, cpx[1] + r], fill=colr, outline=(20, 20, 24))
        geo['handles']['c'] = geo['center']
    return out, geo
