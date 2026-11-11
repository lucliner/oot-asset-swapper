"""Textures in the donor's own block-compressed format, one flat-colour block per source pixel.

The game's textures are block compressed (ASTC for colour and normals, BC4 for single channels). A block that holds one
flat colour is valid in all of them and needs no encoder: a texture of N x M blocks shows an N x M image with hard pixel
edges, in exactly the format the game's materials expect (an uncompressed texture stalled the game's loader).
"""
import struct

import numpy as np

import bntx
from bntx import Brti, DATA_START, block_height, swizzle, swizzled_size

ASTC_BLOCKS = {0x2d: (4, 4), 0x2e: (5, 4), 0x2f: (5, 5), 0x30: (6, 5), 0x31: (6, 6), 0x32: (8, 5), 0x33: (8, 6),
               0x34: (8, 8), 0x35: (10, 5), 0x36: (10, 6), 0x37: (10, 8), 0x38: (10, 10), 0x39: (12, 10), 0x3a: (12, 12)}
BC1, BC4 = 0x1a, 0x1d
VOID_EXTENT = 0xFFFFFFFFFFFFFDFC      # ASTC LDR void-extent block header, no extent given


def flat_blocks(fmt, rgba):
    """Block bytes (rows, cols, bytes per block), block size in pixels and bytes per block, for one flat-colour block
    per source pixel of rgba (rows, cols, 4) uint8 in format `fmt`; None when the format is not one this can write."""
    t = fmt >> 8
    rows, cols = rgba.shape[:2]
    if t in ASTC_BLOCKS:
        blk = np.zeros((rows, cols, 16), np.uint8)
        blk[:, :, :8] = np.frombuffer(struct.pack('<Q', VOID_EXTENT), np.uint8)
        col = rgba.astype(np.uint16) * 257                                   # 8 -> 16 bits (sRGB takes the top 8)
        blk[:, :, 8:] = col.astype('<u2').view(np.uint8).reshape(rows, cols, 8)
        return blk, ASTC_BLOCKS[t], 16
    if t == BC4:
        blk = np.zeros((rows, cols, 8), np.uint8)
        blk[:, :, 0] = blk[:, :, 1] = rgba[:, :, 0]                          # both endpoints equal, indices 0
        return blk, (4, 4), 8
    if t == BC1:
        r, g, b = (rgba[:, :, k].astype(np.uint16) for k in range(3))
        c = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
        blk = np.zeros((rows, cols, 8), np.uint8)
        blk[:, :, 0:2] = c.astype('<u2').view(np.uint8).reshape(rows, cols, 2)
        blk[:, :, 2:4] = blk[:, :, 0:2]
        return blk, (4, 4), 8
    return None


def make_flat(donor, rgba):
    """A texture file from `donor` (decompressed one-texture BNTX) showing rgba (rows, cols, 4) uint8, one flat block
    per pixel, in the donor's own format. Returns None when the donor's format is not one flat_blocks can write."""
    d = Brti(donor)
    fb = flat_blocks(d.fmt, np.ascontiguousarray(rgba, np.uint8))
    if fb is None or donor[0xff0:0xff4] != b'BRTD':
        return None
    blk, (bw, bh_px), _bpb = fb
    rows, cols = blk.shape[:2]
    # the donor's own block height is the one the game and its GPU layer already take for this slot; a smaller image
    # cannot use a taller block than it has rows
    bh = min(1 << d.block_height_log2, block_height(rows))
    pix = swizzle(blk, bh)
    size = (len(pix) + d.alignment - 1) // d.alignment * d.alignment
    pix += bytes(size - len(pix))
    b = bytearray(donor[:0xff0])
    at = d.at
    struct.pack_into('<H', b, at + 0x16, 1)                                   # one mip level
    struct.pack_into('<ii', b, at + 0x24, cols * bw, rows * bh_px)
    struct.pack_into('<I', b, at + 0x34, bh.bit_length() - 1)
    struct.pack_into('<I', b, at + 0x50, size)
    mip_ptr, = struct.unpack_from('<Q', donor, at + 0x70)
    struct.pack_into('<Q', b, mip_ptr, 0)
    rlt = bytearray(donor[donor.find(b'_RLT'):])
    end = (DATA_START + size + 0xfff) // 0x1000 * 0x1000
    b += b'BRTD' + struct.pack('<IQ', 0, size + 0x10)
    b += bytes(DATA_START - len(b)) + pix + bytes(end - DATA_START - size)
    struct.pack_into('<I', rlt, 4, end)
    struct.pack_into('<I', rlt, 0x34, size + 0x10)
    struct.pack_into('<I', b, 0x18, end)
    out = bytearray(bytes(b) + bytes(rlt))
    struct.pack_into('<I', out, 0x1c, len(out))
    return bytes(out)


def make_inplace(donor, rgba, resample=None, detail=False):
    """The donor with its base level replaced by rgba in place: same dimensions, format, block height, sizes and tables
    (only the mip count becomes 1, so the unused lower levels are never read). The image is resampled onto the donor's
    block grid, one flat block per grid cell. Returns None when the donor's format is not one flat_blocks can write.

    Rewriting a file's internal size fields made the game's loader stall; leaving every size alone does not."""
    from PIL import Image
    d = Brti(donor)
    t = d.fmt >> 8
    dims = ASTC_BLOCKS.get(t) or ((4, 4) if t in (BC1, BC4) else None)
    if dims is None:
        return None
    cols, rows = -(-d.w // dims[0]), -(-d.h // dims[1])
    src = np.ascontiguousarray(rgba, np.uint8)
    if detail and (t in ASTC_BLOCKS or t == BC1):
        # a real image: every texel of the donor's size is encoded (ASTC or BC1), not one colour per block
        import block_encode
        full = np.asarray(Image.fromarray(src[..., :3]).resize((cols * dims[0], rows * dims[1]), Image.LANCZOS), np.uint8)
        blk = (block_encode.encode_astc_best(full, dims[0], dims[1]) if t in ASTC_BLOCKS else block_encode.encode_bc1(full))
    else:
        if src.shape[:2] != (rows, cols):
            small = src.shape[0] * src.shape[1] <= 128 * 128
            mode = resample if resample is not None else (Image.NEAREST if small else Image.LANCZOS)
            src = np.asarray(Image.fromarray(src).resize((cols, rows), mode), np.uint8)
        blk = flat_blocks(d.fmt, src)[0]
    out = bytearray(donor)
    levels = _mip_layout(donor, d, dims)
    if levels is None:
        # (no readable mip table: only the base level is replaced, and the unused lower levels are declared away)
        bh = 1 << d.block_height_log2
        pix = swizzle(blk, bh)
        if len(pix) > d.image_size:
            return None
        out[DATA_START:DATA_START + len(pix)] = pix
        struct.pack_into('<H', out, d.at + 0x16, 1)
        return bytes(out)
    # every level of the donor's mip chain is rewritten in place (same offsets, sizes and block heights): a level left as the
    # donor had it would show Link's own texture wherever the game samples a lower level (shrunk or steeply seen surfaces)
    for k, (off, size, bh, lw, lh) in enumerate(levels):
        if k == 0:
            data = blk
        else:
            gw, gh = -(-lw // dims[0]), -(-lh // dims[1])
            if detail and (t in ASTC_BLOCKS or t == BC1):
                import block_encode
                lvl = np.asarray(Image.fromarray(src[..., :3]).resize((gw * dims[0], gh * dims[1]), Image.LANCZOS), np.uint8)
                data = (block_encode.encode_astc_best(lvl, dims[0], dims[1]) if t in ASTC_BLOCKS else block_encode.encode_bc1(lvl))
            else:
                small = np.asarray(Image.fromarray(src).resize((gw, gh), Image.BILINEAR if src.shape[0] * src.shape[1] > 16384 else Image.NEAREST), np.uint8)
                data = flat_blocks(d.fmt, small)[0]
        pix = swizzle(data, bh)
        if len(pix) > size:
            return None
        out[DATA_START + off:DATA_START + off + len(pix)] = pix
    return bytes(out)


def _mip_layout(donor, d, dims):
    """[(offset, byte size, block height in GOBs, width, height)] of the donor's mip levels, read from its own mip offset
    table; None when the table does not describe a chain this code can rewrite."""
    try:
        n = d.mips
        ptr, = struct.unpack_from('<Q', donor, d.at + 0x70)
        offs = [o - DATA_START for o in struct.unpack_from('<%dQ' % n, donor, ptr)]       # (absolute file offsets)
        if n < 1 or offs[0] != 0:
            return None
        ends = offs[1:] + [d.image_size]
        out = []
        for k in range(n):
            lw, lh = max(1, d.w >> k), max(1, d.h >> k)
            gw, gh = -(-lw // dims[0]), -(-lh // dims[1])
            room = ends[k] - offs[k]
            out.append((offs[k], room, lw, lh, gw, gh))
        res = []
        bpp = 16 if (d.fmt >> 8) in ASTC_BLOCKS else 8
        for (off, room, lw, lh, gw, gh) in out:
            cand = [b for b in (1, 2, 4, 8, 16) if swizzled_size(gw, gh, bpp, b) <= room]
            if not cand:
                return None
            # the block height the level really uses: the largest one the game's tools would pick that still fits
            want = 1
            while want < gh // 8 and want < (1 << d.block_height_log2):
                want *= 2
            bh = want if want in cand else max(cand)
            res.append((off, room, bh, lw, lh))
        return res
    except Exception:
        return None
