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
    decompressed size."""
    import os
    path = os.path.join(mod_dir, 'romfs', *TABLE_REL.split('/'))
    table = Restbl(open(path, 'rb').read() and rom.decompress(open(path, 'rb').read()) if os.path.exists(path) else rom.read(TABLE_REL))
    for rel, size in changed.items():
        old_entry = table.get(rel)
        if old_entry is None:
            extra = TEXTURE_EXTRA
        elif rel.startswith('Tex/'):
            extra = TEXTURE_EXTRA
        else:
            extra = old_entry - len(rom.read(rel))          # the parsed-size surplus of the original file
        table.set(rel, size + extra)
        log('  size table: %s %s -> %d' % (rel, old_entry, size + extra))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dict_name = rom.dict_of(TABLE_REL)
    open(path, 'wb').write(rom.compress(table.to_bytes(), dict_name=dict_name))
