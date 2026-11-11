"""The game's texture files (`Tex/<Name>.bntx.zs`, one BNTX each, plain zstd without a dictionary).

A new texture is made from a donor file of the game: its metadata block (names, dictionary, the BRTI description and the
relocation table) is kept, and the image fields, the pixel data and the section sizes are rewritten. Pixels are written
uncompressed (RGBA8): the texture's own description tells the game what it holds, so any size and format works.

Pixel layout is the Tegra X1 block-linear swizzle: 64-byte x 8-row GOBs, stacked `block_height` high.
"""
import struct

import numpy as np

FMT_RGBA8 = 0x0b01
FMT_RGBA8_SRGB = 0x0b06
DATA_START = 0x1000          # pixel data follows the 16-byte BRTD header at 0xff0, in every texture of the game


def block_height(rows):
    """Block height (in GOBs) the game's tools pick for an image `rows` tall."""
    bh = 1
    while bh < rows // 8 and bh < 16:
        bh *= 2
    return bh


def _gob_addresses(w_bytes, rows, bh):
    """Byte offset in the swizzled surface of every (row, byte column): (rows, w_bytes) int64."""
    y = np.arange(rows)[:, None]
    x = np.arange(w_bytes)[None, :]
    gobs_per_row = (w_bytes + 63) // 64
    gob = ((y // (8 * bh)) * 512 * bh * gobs_per_row + (x // 64) * 512 * bh + ((y % (8 * bh)) // 8) * 512)
    return (gob + ((x % 64) // 32) * 256 + ((y % 8) // 2) * 64 + ((x % 32) // 16) * 32 + (y % 2) * 16 + (x % 16))


def swizzled_size(w, h, bpp, bh):
    pitch = (w * bpp + 63) // 64 * 64
    rows = (h + 8 * bh - 1) // (8 * bh) * (8 * bh)
    return pitch * rows


def swizzle(pixels, bh):
    """pixels (h, w, bpp) uint8 -> block-linear bytes."""
    h, w, bpp = pixels.shape
    out = np.zeros(swizzled_size(w, h, bpp, bh), np.uint8)
    addr = _gob_addresses(w * bpp, h, bh)
    out[addr] = pixels.reshape(h, w * bpp)
    return out.tobytes()


def deswizzle(data, w, h, bpp, bh):
    buf = np.frombuffer(data, np.uint8)
    addr = _gob_addresses(w * bpp, h, bh)
    return buf[addr].reshape(h, w, bpp)


class Brti:
    """The fields of a texture's BRTI block that matter here (offsets within the block)."""
    def __init__(self, b, at=None):
        at = b.find(b'BRTI') if at is None else at      # (0x200, or 0x208 when the texture's name is longer)
        if at < 0 or b.find(b'BRTI', at + 4) >= 0:
            raise ValueError('expected exactly one BRTI block (a one-texture BNTX)')
        self.at = at
        self.mips, = struct.unpack_from('<H', b, at + 0x16)
        self.fmt, self.usage, self.w, self.h = struct.unpack_from('<IIii', b, at + 0x1c)
        self.block_height_log2, = struct.unpack_from('<I', b, at + 0x34)
        self.image_size, self.alignment = struct.unpack_from('<II', b, at + 0x50)
        self.channels = tuple(b[at + 0x58:at + 0x5c])


def read(b):
    return Brti(b)


def make(donor, rgba, srgb=False, channels=None):
    """A texture file from `donor` (decompressed BNTX bytes) with new pixels: rgba (h, w, 4) uint8. `channels` is the
    donor's channel mapping unless given (the numbers: 0 zero, 1 one, 2 R, 3 G, 4 B, 5 A)."""
    d = Brti(donor)
    if donor[0xff0:0xff4] != b'BRTD':
        raise ValueError('donor is not a one-texture BNTX of the game\'s layout')
    h, w = rgba.shape[:2]
    bh = block_height(h)
    pix = swizzle(np.ascontiguousarray(rgba, np.uint8), bh)
    size = (len(pix) + d.alignment - 1) // d.alignment * d.alignment
    pix += b'\0' * (size - len(pix))
    b = bytearray(donor[:0xff0])
    at = d.at
    struct.pack_into('<H', b, at + 0x16, 1)                                   # one mip level
    struct.pack_into('<I', b, at + 0x1c, FMT_RGBA8_SRGB if srgb else FMT_RGBA8)
    struct.pack_into('<ii', b, at + 0x24, w, h)
    struct.pack_into('<I', b, at + 0x34, bh.bit_length() - 1)
    struct.pack_into('<I', b, at + 0x50, size)
    b[at + 0x58:at + 0x5c] = bytes(channels if channels is not None else d.channels)
    mip_ptr, = struct.unpack_from('<Q', donor, at + 0x70)
    struct.pack_into('<Q', b, mip_ptr, 0)                                     # level 0 starts at the data
    rlt = bytearray(donor[donor.find(b'_RLT'):])
    end = (DATA_START + size + 0xfff) // 0x1000 * 0x1000
    b += b'BRTD' + struct.pack('<IQ', 0, size + 0x10)
    b += b'\0' * (DATA_START - len(b)) + pix + b'\0' * (end - DATA_START - size)
    struct.pack_into('<I', rlt, 4, end)                                       # the table's own position
    struct.pack_into('<I', rlt, 0x34, size + 0x10)                            # section 1 (the data): size
    struct.pack_into('<I', b, 0x18, end)                                      # header: relocation table, file size
    out = bytearray(bytes(b) + bytes(rlt))
    struct.pack_into('<I', out, 0x1c, len(out))
    return bytes(out)
