"""The game's file containers: zstd with the game's trained dictionaries, and SARC.

Everything is read from the user's own romfs; nothing here carries game data.
"""
import os
import struct

import zstandard


def sarc_read(b):
    """SARC archive -> {name: bytes}. Entries without a name are keyed by their hash, as 8 hex digits."""
    if b[:4] != b'SARC':
        raise ValueError('not a SARC archive')
    data_off = struct.unpack_from('<I', b, 0x0c)[0]
    if b[0x14:0x18] != b'SFAT':
        raise ValueError('SARC: SFAT missing')
    cnt = struct.unpack_from('<H', b, 0x1a)[0]
    name_base = 0x20 + cnt * 16 + 8
    out = {}
    for i in range(cnt):
        h, attr, s, e = struct.unpack_from('<IIII', b, 0x20 + i * 16)
        if attr >> 24:
            no = name_base + (attr & 0xffffff) * 4
            nm = b[no:b.index(b'\0', no)].decode()
        else:
            nm = '%08x' % h
        out[nm] = b[data_off + s:data_off + e]
    return out


class Romfs:
    """A romfs folder: reads and writes the game's `.zs` files with the right dictionary."""

    def __init__(self, root):
        self.root = root
        raw = open(os.path.join(root, 'Pack', 'ZsDic.pack.zs'), 'rb').read()
        files = sarc_read(zstandard.ZstdDecompressor().decompressobj().decompress(raw))
        self.dicts = {}                       # dict id -> (name, ZstdCompressionDict)
        for name, data in files.items():
            d = zstandard.ZstdCompressionDict(data)
            self.dicts[d.dict_id()] = (name, d)

    def path(self, rel):
        return os.path.join(self.root, *rel.split('/'))

    def decompress(self, raw):
        did = zstandard.get_frame_parameters(raw).dict_id
        if did:
            if did not in self.dicts:
                raise ValueError('zstd dictionary %d is not in ZsDic.pack' % did)
            dec = zstandard.ZstdDecompressor(dict_data=self.dicts[did][1])
        else:
            dec = zstandard.ZstdDecompressor()
        return dec.decompressobj().decompress(raw)

    def read(self, rel):
        """The decompressed bytes of a romfs file ('Model/X.bfres.zs'); a file that is not .zs is read as it is."""
        raw = open(self.path(rel), 'rb').read()
        return self.decompress(raw) if rel.endswith('.zs') else raw

    def compress(self, data, dict_name='zs.zsdic', level=19):
        """Compress as the game does: one frame with a dictionary (the default `zs.zsdic` is the one the Model files use;
        None for a plain frame, as the Tex files have)."""
        if dict_name is None:
            return zstandard.ZstdCompressor(level=level).compress(data)
        for _id, (name, d) in self.dicts.items():
            if name == dict_name:
                return zstandard.ZstdCompressor(level=level, dict_data=d).compress(data)
        raise ValueError('no dictionary named ' + dict_name)

    def dict_of(self, rel):
        """Name of the dictionary a romfs file was compressed with (or None)."""
        with open(self.path(rel), 'rb') as f:
            did = zstandard.get_frame_parameters(f.read(32)).dict_id
        return self.dicts[did][0] if did in self.dicts else None
