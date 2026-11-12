"""The game's resource size table (`System/Resource/ResourceSizeTable.Product.111.Nin_Ounce_NVN2.rsizetable.zs`).

It maps the CRC32 of a resource's path (without `.zs`) to the memory the engine reserves for it: for textures the
decompressed size + 0x118, for models the size of the parsed file. A mod that changes a file's size has to change its
entry too, or the engine reserves the wrong amount (it stalls on the loading screen).
Layout: 'RESTBL', u32 version, u32 name size, u32 hash count, u32 collision count, then (crc, size) pairs sorted by crc,
then the collision entries (name + size), which are kept as they are.
"""
import struct
import zlib

TABLE_REL = 'System/Resource/ResourceSizeTable.Product.111.Nin_Ounce_NVN2.rsizetable.zs'
TEXTURE_EXTRA = 0x118


class Restbl:
    def __init__(self, data):
        if data[:6] != b'RESTBL':
            raise ValueError('not a RESTBL')
        self.version, self.name_size, nh, self.n_collisions = struct.unpack_from('<IIII', data, 6)
        self.entries = {}
        for i in range(nh):
            h, s = struct.unpack_from('<II', data, 22 + i * 8)
            self.entries[h] = s
        self.tail = data[22 + nh * 8:]

    @staticmethod
    def key(path):
        """CRC32 of a resource path as the table uses it: no '.zs'."""
        return zlib.crc32(path[:-3].encode() if path.endswith('.zs') else path.encode())

    def get(self, path):
        return self.entries.get(self.key(path))

    def set(self, path, size):
        self.entries[self.key(path)] = size

    def to_bytes(self):
        out = bytearray(b'RESTBL' + struct.pack('<IIII', self.version, self.name_size, len(self.entries), self.n_collisions))
        for h in sorted(self.entries):
            out += struct.pack('<II', h, self.entries[h])
        return bytes(out) + self.tail


def patch(rom, mod_dir, changed, log=print):
    """Write the mod's copy of the size table with the entries of `changed` {romfs path ('Tex/X.bntx.zs'): new
    decompressed size} set. Textures get size + 0x118; any other file keeps the original table's extra over its old
    decompressed size.

    The game uses one mod's table only (the mod whose folder name sorts last), so the tables of the other enabled mods in the
    same mods folder are merged in first: their grown entries stay in force beside this mod's."""
    import os
    path = os.path.join(mod_dir, 'romfs', *TABLE_REL.split('/'))
    base = Restbl(rom.read(TABLE_REL))
    table = Restbl(rom.read(TABLE_REL))
    mods_dir = os.path.dirname(os.path.abspath(mod_dir))
    me = os.path.basename(os.path.abspath(mod_dir))
    later = []
    try:
        others = sorted(d for d in os.listdir(mods_dir) if d != me and not d.startswith(('-', '_')) and os.path.isdir(os.path.join(mods_dir, d)))
    except OSError:
        others = []
    for d in others:
        t = os.path.join(mods_dir, d, 'romfs', *TABLE_REL.split('/'))
        if not os.path.exists(t):
            continue
        try:
            theirs = Restbl(rom.decompress(open(t, 'rb').read()))
        except Exception:
            continue
        grown = {h: v for h, v in theirs.entries.items() if base.entries.get(h) != v}
        table.entries.update(grown)
        if grown:
            log('  size table: %d entries of the mod "%s" kept' % (len(grown), d))
        if d > me:
            later.append(d)
    if os.path.exists(path):                                      # this mod's earlier builds (other targets) stay too
        try:
            mine = Restbl(rom.decompress(open(path, 'rb').read()))
            table.entries.update({h: v for h, v in mine.entries.items() if base.entries.get(h) != v})
        except Exception:
            pass
    for rel, size in changed.items():
        old_entry = base.get(rel)
        if old_entry is None or rel.startswith('Tex/'):
            extra = TEXTURE_EXTRA
        else:
            extra = old_entry - len(rom.read(rel))          # the parsed-size surplus of the original file
            frac = float(os.environ.get('ASSETSWAP_RESTBL_MARGIN', '1.0'))     # (for experiments: another margin)
            # and room to spare. The game reserves about 2.5 x the metadata plus the buffers and a little more; with two swapped
            # models loaded together (a Zora and the King, say) a 12% margin left one of them undrawn, a margin of the model's
            # own size did not. The surplus costs only reserved memory, so it is capped.
            extra += min(0x400000, max(0x10000, int(size * frac)))
        table.set(rel, size + extra)                        # the buffers, and short by a byte means the model is not shown
        log('  size table: %s %s -> %d' % (rel, old_entry, size + extra))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'wb').write(rom.compress(table.to_bytes(), dict_name=rom.dict_of(TABLE_REL)))
    if later:
        log('  NOTE: the game takes the size table of the mod whose folder sorts last, and "%s" sorts after this one: '
            'its table lacks the entries of this mod. Rename this mod to sort last (e.g. "zz-%s"), or rebuild that mod so its table picks these up.' % (later[-1], me))
