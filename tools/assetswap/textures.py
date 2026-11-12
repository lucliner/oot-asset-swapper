"""Textures for a swap: the material of each shape that carries an imported part names its textures (`<Material>_<Model>_Alb`,
`_Nrm`, `_Rgh`, `_AO`, ...), one file each in the game's `Tex` folder. A mod replaces those files: the albedo with the
imported model's image, the other maps with neutral values (a flat normal map, mid roughness, no occlusion, no metal)."""
import os
import struct

import numpy as np
from PIL import Image

import bntx
import bntx_flat

# what each kind of map holds where the shader reads it (r, g, b, a): the neutral value of the map
NEUTRAL = {
    'Nrm': (0.5, 0.5, 1.0, 1.0),     # a flat normal map (the donors keep x and y in their first two channels)
    'Rgh': (0.75,) * 4,              # mostly matte
    'AO': (1.0,) * 4,                # nothing occluded
    'Mtl': (0.0,) * 4,               # not metal
    'Emm': (0.0,) * 4,               # not glowing
}


def material_textures(f, mat_index):
    """Texture names of a material of a Bfres (FMAT: name array at +0x20, texture count at +0xa2)."""
    import struct as st
    b = f.b
    mat_arr = st.unpack_from('<Q', b, f.model_off + 0x38)[0]
    o = mat_arr + mat_index * 0xb0
    arr = st.unpack_from('<Q', b, o + 0x20)[0]
    n = b[o + 0xa2]
    return [f._str(st.unpack_from('<Q', b, arr + i * 8)[0]) for i in range(n)]


def own_albedo(rom, f, mat_index, cache={}):
    """True when the material's base-colour texture belongs to this model alone (so a mod can replace it) and is in a
    format that can be written in place."""
    key = (f.model_name, mat_index)
    if key not in cache:
        ok = False
        for n in material_textures(f, mat_index):
            if kind_of(n) == 'Alb' and f.model_name in n and os.path.exists(rom.path('Tex/%s.bntx.zs' % n)):
                t = bntx.read(rom.read('Tex/%s.bntx.zs' % n)).fmt >> 8
                ok = t in bntx_flat.ASTC_BLOCKS or t in (bntx_flat.BC1, bntx_flat.BC4)
                break
        cache[key] = ok
    return cache[key]


def albedo_name(f, mat_index):
    """Name of the material's base-colour texture (None when it has none)."""
    return next((n for n in material_textures(f, mat_index) if kind_of(n) == 'Alb'), None)


KINDS = ('Alb', 'Nrm', 'Rgh', 'AO', 'Mtl', 'Emm', 'Thc', 'Gn2', 'Gn3', 'RAH', 'Emc', 'Fzm', 'Fzn')


def kind_of(name):
    """The map kind a texture name ends in. A colour variant carries a number after it (`..._Alb.00`, `.01`: the game picks one
    per character), which does not change the kind."""
    suffix = name.rsplit('_', 1)[-1].split('.')[0]
    return suffix if suffix in KINDS else None


def variants(rom, name):
    """Names of the texture and of every numbered colour variant of it that is in the romfs (`X_Alb.00` -> X_Alb.00, .01, .02)."""
    base, dot, num = name.rpartition('.')
    if not dot or not num.isdigit():
        return [name]
    folder = os.path.dirname(rom.path('Tex/%s.bntx.zs' % name))
    try:
        found = sorted(fn[:-len('.bntx.zs')] for fn in os.listdir(folder)
                       if fn.startswith(base + '.') and fn.endswith('.bntx.zs') and fn[len(base) + 1:-len('.bntx.zs')].isdigit())
    except OSError:
        found = []
    return found or [name]


def median_pixel(donor):
    """One RGBA pixel holding the median of the donor texture's own pixels: a map whose layout this tool does not know is set to
    its average, so it keeps its overall effect (a little shine, a little gloss) without printing its old layout on a new model."""
    import gametex
    d = bntx.Brti(donor)
    im = gametex.decode(donor, 64, rgba=True)
    if im is None:
        return None
    return np.median(im.reshape(-1, 4), axis=0).astype(np.uint8).reshape(1, 1, 4)


# Link's skin carries two more maps laid out on his own UVs (thickness and a gradient map): left alone they print his
# layout over an imported model's face as blocks and stripes. One flat colour, the median of Link's own map.
FLAT_MAPS = {'Thc': (14, 0, 0, 255), 'Gn2': (178, 216, 0, 255)}


def neutral_pixels(kind, channels, size=4):
    """RGBA8 pixels that read as the kind's neutral value through the donor's channel mapping (0 zero, 1 one, 2..5 R G B A)."""
    px = np.zeros(4)
    for shader_channel, src in enumerate(channels):
        if src >= 2:
            px[src - 2] = NEUTRAL[kind][shader_channel]
    return np.broadcast_to(np.rint(px * 255).astype(np.uint8), (size, size, 4)).copy()


def pad_unused(rgb, uv, tris, max_side=2048):
    """Fill the parts of an atlas that no triangle uses with the colours of the nearest used area ("padding").
    Block compression mixes the colours inside a block: without padding, the dark or white background between islands
    bleeds into blocks that straddle an island's edge and shows up as odd patches along seams.
    rgb (h, w, 3) uint8; uv (n, 2) in 0..1, v down as the game stores it; tris (m, 3)."""
    from PIL import Image, ImageDraw
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    mw, mh = max(1, int(w * scale)), max(1, int(h * scale))
    mask_img = Image.new('L', (mw, mh), 0)
    dr = ImageDraw.Draw(mask_img)
    pts = np.asarray(uv, np.float64)[np.asarray(tris)] * np.array([mw, mh])
    for t in pts:
        dr.polygon([tuple(p) for p in t], fill=255)
    mask = np.asarray(mask_img) > 0
    if mask.all() or not mask.any():
        return rgb
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    # grow the used area a little first (the polygon fill misses the half-covered pixels along an edge)
    grown = mask.copy()
    for _ in range(2):
        g = grown.copy()
        g[1:] |= grown[:-1]; g[:-1] |= grown[1:]; g[:, 1:] |= grown[:, :-1]; g[:, :-1] |= grown[:, 1:]
        grown = g
    # push-pull: average the used colours into coarser and coarser levels, then fill the holes from the coarse side
    levels = [(rgb.astype(np.float64) * mask[..., None], mask.astype(np.float64))]
    while min(levels[-1][1].shape) > 2:
        c, m = levels[-1]
        hh, ww = (c.shape[0] // 2) * 2, (c.shape[1] // 2) * 2
        c2 = c[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2, 3).sum((1, 3))
        m2 = m[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2).sum((1, 3))
        levels.append((c2, m2))
    c, m = levels[-1]
    fill = c / np.maximum(m, 1e-9)[..., None]
    for c, m in reversed(levels[:-1]):
        up = np.repeat(np.repeat(fill, 2, 0), 2, 1)
        up = np.pad(up, ((0, c.shape[0] - up.shape[0]), (0, c.shape[1] - up.shape[1]), (0, 0)), mode='edge')
        own = c / np.maximum(m, 1e-9)[..., None]
        fill = np.where((m > 0)[..., None], own, up)
    out = rgb.copy()
    hole = ~mask
    out[hole] = np.clip(np.rint(fill[hole]), 0, 255).astype(np.uint8)
    return out


def load_image(path):
    im = Image.open(path)
    if im.mode == 'P' and 'transparency' in im.info:
        im = im.convert('RGBA')
    return np.asarray(im.convert('RGBA'), np.uint8)


def build(rom, f, slot_textures, log=print, geometry=None):
    """{romfs path: compressed file bytes} for every texture of the materials of the given shapes.
    slot_textures: {shape index: image path or None}."""
    out = {}
    todo = []
    for si, img_path in slot_textures.items():
        mat = f.shapes[si].material
        for name0 in material_textures(f, mat):
            kind = kind_of(name0)
            if kind is None or not os.path.exists(rom.path('Tex/%s.bntx.zs' % name0)):
                log('  texture %-40s left as it is (%s)' % (name0, 'unknown kind' if kind is None else 'not in the romfs'))
                continue
            if f.model_name not in name0:
                # a texture shared with other models (cloth tiles, detail maps) is not this model's to change
                log('  texture %-40s left as it is (shared with other models)' % name0)
                continue
            for name in variants(rom, name0):
                todo.append((si, img_path, name, kind))
    for si, img_path, name, kind in todo:
        if True:
            rel = 'Tex/%s.bntx.zs' % name
            if rel in out:
                continue
            donor = rom.read(rel)
            d = bntx.read(donor)
            if kind == 'Alb':
                if not img_path:
                    continue
                pix = load_image(img_path)             # (resampled to the donor's size by make_inplace)
                if geometry and si in geometry and geometry[si][0] is not None:
                    pix = pix.copy()
                    pix[..., :3] = pad_unused(pix[..., :3], *geometry[si])    # no island-edge blocks full of background
            elif kind in FLAT_MAPS:
                pix = np.broadcast_to(np.array(FLAT_MAPS[kind], np.uint8), (1, 1, 4)).copy()
            elif kind in NEUTRAL:
                pix = neutral_pixels(kind, d.channels, 1)
            else:
                pix = median_pixel(donor)
                if pix is None:
                    log('  texture %-40s left as it is (cannot be read)' % name)
                    continue
            new = bntx_flat.make_inplace(donor, pix, detail=(kind == 'Alb'))
            if new is None:
                log('  texture %-40s: format 0x%04x cannot be written in place; left as it is' % (name, d.fmt))
                continue
            out[rel] = rom.compress(new, dict_name=rom.dict_of(rel))
            log('  texture %-40s <- %s' % (name, os.path.basename(img_path) if kind == 'Alb' else 'neutral %s' % kind))
    return out
