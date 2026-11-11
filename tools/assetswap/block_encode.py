"""Block encoders for the game's colour textures: BC1 and ASTC (LDR, RGB), vectorised with numpy.

BC1: per 4x4 block two RGB565 endpoints (along the block's principal colour axis) and 2-bit indices.

ASTC: one partition, colour endpoint mode 8 (RGB, direct) with 8-bit endpoints, and a weight grid smaller than or equal to the
block, weights of 1, 2 or 3 bits. The layout is chosen so that the endpoints are the full 8 bits (weights take at most 63 of the
128 bits), which keeps every integer sequence a plain bit field (no trits or quints). The weight grid is fitted to the texels by
least squares through ASTC's own bilinear infill, then the endpoints are refitted to the quantised weights.
Output is checked against an independent decoder in tests/test_assetswap.py (when it is installed).
"""
import numpy as np

# -- helpers ------------------------------------------------------------------------------------------------------------


def to_blocks(img, bw, bh):
    """img (H, W, C) with H, W multiples of bh, bw -> (rows, cols, bh, bw, C)."""
    h, w, c = img.shape
    return img.reshape(h // bh, bh, w // bw, bw, c).transpose(0, 2, 1, 3, 4)


def _pca_endpoints(X):
    """X (n, k, 3) float -> (e0, e1 (n, 3), t (n, k) in 0..1) endpoints along the principal axis at the extreme projections."""
    mean = X.mean(1)
    Xc = X - mean[:, None, :]
    cov = np.einsum('nki,nkj->nij', Xc, Xc)
    _w, v = np.linalg.eigh(cov)
    axis = v[:, :, -1]                                           # largest eigenvalue
    proj = np.einsum('nki,ni->nk', Xc, axis)
    lo, hi = proj.min(1), proj.max(1)
    e0 = mean + lo[:, None] * axis
    e1 = mean + hi[:, None] * axis
    span = np.where(hi - lo < 1e-6, 1.0, hi - lo)
    t = (proj - lo[:, None]) / span[:, None]
    t[(hi - lo) < 1e-6] = 0.0
    return e0, e1, t


# -- BC1 ---------------------------------------------------------------------------------------------------------------


def _expand565(c):
    r, g, b = (c >> 11) & 31, (c >> 5) & 63, c & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.float64)


def encode_bc1(img):
    """img (H, W, 3) uint8, H and W multiples of 4 -> (rows, cols, 8) uint8 blocks."""
    B = to_blocks(img, 4, 4)
    rows, cols = B.shape[:2]
    X = B.reshape(rows * cols, 16, 3).astype(np.float64)
    e0, e1, _t = _pca_endpoints(X)

    def q565(e):
        e = np.clip(np.rint(e), 0, 255).astype(np.int64)
        return ((e[:, 0] >> 3) << 11) | ((e[:, 1] >> 2) << 5) | (e[:, 2] >> 3)
    c0, c1 = q565(e0), q565(e1)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)         # c0 > c1: the four-colour mode
    solid = c0 == c1
    p0, p1 = _expand565(c0), _expand565(c1)
    pal = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], 1)           # (n, 4, 3)
    d = ((X[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)                  # (n, 16, 4)
    idx = d.argmin(-1)
    idx[solid] = 0
    bits = np.zeros(len(idx), np.uint64)
    for k in range(16):
        bits |= idx[:, k].astype(np.uint64) << np.uint64(2 * k)
    out = np.zeros((len(idx), 8), np.uint8)
    out[:, 0], out[:, 1] = c0 & 255, c0 >> 8
    out[:, 2], out[:, 3] = c1 & 255, c1 >> 8
    for k in range(4):
        out[:, 4 + k] = ((bits >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    return out.reshape(rows, cols, 8)


# -- ASTC --------------------------------------------------------------------------------------------------------------

# weight grids the 2D block mode field can name: (layout, A, B) -> (width, height)
def _legal_grids(bw, bh):
    out = []
    for layout in (0, 1, 2):
        for a in range(4):
            for b in range(4):
                w, h = {0: (b + 4, a + 2), 1: (b + 8, a + 2), 2: (a + 2, b + 8)}[layout]
                if w <= bw and h <= bh:
                    out.append((w, h, layout, a, b))
    return out


# weight precision: bits -> (levels, base quant mode, unquantised values 0..64)
_PREC = {1: (2, 2, [0, 64]), 2: (4, 4, [0, 21, 43, 64]), 3: (8, 7, [0, 9, 18, 27, 37, 46, 55, 64])}


def _block_mode(layout, a, b, qbits):
    base = _PREC[qbits][1]
    return (base >> 1) | (layout << 2) | ((base & 1) << 4) | (a << 5) | (b << 7)       # H = 0, D = 0


def _infill_matrix(bw, bh, wg, hg):
    """(bw*bh, wg*hg) matrix: ASTC's bilinear infill of a weight grid onto the block's texels (weights 0..1)."""
    ds = (1024 + bw // 2) // (bw - 1) if bw > 1 else 0
    dt = (1024 + bh // 2) // (bh - 1) if bh > 1 else 0
    U = np.zeros((bw * bh, wg * hg))
    for t in range(bh):
        for s in range(bw):
            gs = (ds * s * (wg - 1) + 32) >> 6
            gt = (dt * t * (hg - 1) + 32) >> 6
            js, jt, fs, ft = gs >> 4, gt >> 4, gs & 15, gt & 15
            w11 = (fs * ft + 8) >> 4
            w10, w01 = ft - w11, fs - w11
            w00 = 16 - fs - ft + w11
            v0 = js + jt * wg
            for off, wt in ((0, w00), (1, w01), (wg, w10), (wg + 1, w11)):
                if wt:
                    U[t * bw + s, min(v0 + off, wg * hg - 1)] += wt / 16.0
    return U


def best_config(bw, bh, qbits_options=(1, 2)):
    """The legal grid and weight precision with the most weights that fit in 63 bits (8-bit endpoints stay valid)."""
    best = None
    for qb in qbits_options:
        for (w, h, layout, a, b) in _legal_grids(bw, bh):
            n = w * h
            if 24 <= n * qb <= 63:
                key = (n * (qb ** 0.5), n)
                if best is None or key > best[0]:
                    best = (key, (w, h, layout, a, b, qb))
    return best[1]


def encode_astc(img, bw, bh, config=None, with_error=False):
    """img (H, W, 3) uint8, H and W multiples of bh, bw -> (rows, cols, 16) uint8 blocks (and the squared error of each
    block when with_error)."""
    wg, hg, layout, a, b, qb = config or best_config(bw, bh)
    levels, _base, unq = _PREC[qb]
    unq = np.array(unq, np.float64) / 64.0
    B = to_blocks(img, bw, bh)
    rows, cols = B.shape[:2]
    n = rows * cols
    X = B.reshape(n, bw * bh, 3).astype(np.float64)
    e0, e1, t = _pca_endpoints(X)
    U = _infill_matrix(bw, bh, wg, hg)
    P = np.clip(t @ np.linalg.pinv(U).T, 0.0, 1.0)                       # grid weights, least squares through the infill
    L = np.abs(P[:, :, None] - unq[None, None, :]).argmin(-1)           # nearest quantised level
    for _ in range(2):                                                   # refit endpoints to the quantised weights
        alpha = (unq[L] @ U.T)                                           # (n, texels)
        s11 = (alpha * alpha).sum(1); s10 = (alpha * (1 - alpha)).sum(1); s00 = ((1 - alpha) ** 2).sum(1)
        det = s00 * s11 - s10 * s10
        ok = np.abs(det) > 1e-9
        r0 = np.einsum('nk,nkc->nc', 1 - alpha, X)
        r1 = np.einsum('nk,nkc->nc', alpha, X)
        n0 = np.where(ok[:, None], (s11[:, None] * r0 - s10[:, None] * r1) / np.where(ok, det, 1)[:, None], X.mean(1))
        n1 = np.where(ok[:, None], (s00[:, None] * r1 - s10[:, None] * r0) / np.where(ok, det, 1)[:, None], X.mean(1))
        e0, e1 = np.clip(n0, 0, 255), np.clip(n1, 0, 255)
        # re-fit the weights to the new endpoints
        d = e1 - e0
        dd = (d * d).sum(1)
        tt = np.einsum('nkc,nc->nk', X - e0[:, None, :], d) / np.where(dd < 1e-6, 1.0, dd)[:, None]
        tt[dd < 1e-6] = 0.0
        P = np.clip(np.clip(tt, 0, 1) @ np.linalg.pinv(U).T, 0.0, 1.0)
        L = np.abs(P[:, :, None] - unq[None, None, :]).argmin(-1)
    E0 = np.clip(np.rint(e0), 0, 255).astype(np.int64)
    E1 = np.clip(np.rint(e1), 0, 255).astype(np.int64)
    # the decoder takes the endpoints as given only when sum(e1) >= sum(e0): otherwise it contracts blue; mirror the block
    flip = E1.sum(1) < E0.sum(1)
    E0, E1 = np.where(flip[:, None], E1, E0), np.where(flip[:, None], E0, E1)
    L = np.where(flip[:, None], levels - 1 - L, L)
    bits = np.zeros((n, 128), np.uint8)
    mode = _block_mode(layout, a, b, qb)
    for k in range(11):
        bits[:, k] = (mode >> k) & 1
    for k in range(4):                                                    # CEM 8 in bits 13..16 (one partition)
        bits[:, 13 + k] = (8 >> k) & 1
    vals = np.stack([E0[:, 0], E1[:, 0], E0[:, 1], E1[:, 1], E0[:, 2], E1[:, 2]], 1)      # r0 r1 g0 g1 b0 b1
    pos = 17
    for v in range(6):
        for k in range(8):
            bits[:, pos + k] = (vals[:, v] >> k) & 1
        pos += 8
    for i in range(wg * hg):                                              # weights are written backwards from bit 127
        for k in range(qb):
            bits[:, 127 - (i * qb + k)] = (L[:, i] >> k) & 1
    out = np.packbits(bits, axis=1, bitorder='little').reshape(rows, cols, 16)
    if not with_error:
        return out
    alpha = unq[L] @ U.T
    recon = E0[:, None, :] * (1 - alpha[:, :, None]) + E1[:, None, :] * alpha[:, :, None]
    return out, ((X - recon) ** 2).sum((1, 2))


def candidate_configs(bw, bh, per_precision=3):
    """The legal weight grids worth trying: for each weight precision, the ones with the most weights."""
    out = []
    for qb in (1, 2, 3):
        grids = [(w * h, w, h, layout, a, b) for (w, h, layout, a, b) in _legal_grids(bw, bh) if 24 <= w * h * qb <= 63]
        grids.sort(reverse=True)
        out += [(w, h, layout, a, b, qb) for (_n, w, h, layout, a, b) in grids[:per_precision]]
    return out


def encode_astc_best(img, bw, bh):
    """Each block is encoded with every candidate layout and keeps the one that reconstructs it best (a block mode is
    stored per block, so the layouts can differ from block to block)."""
    best, best_err = None, None
    for cfg in candidate_configs(bw, bh):
        out, err = encode_astc(img, bw, bh, cfg, with_error=True)
        if best is None:
            best, best_err = out, err
            continue
        better = (err < best_err).reshape(best.shape[:2])
        best = np.where(better[..., None], out, best)
        best_err = np.where(better.reshape(-1), err, best_err)
    return best
