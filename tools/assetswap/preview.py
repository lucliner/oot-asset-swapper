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
