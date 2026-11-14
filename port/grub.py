"""Turn 10 ForzaTech "Grub" bundles (.modelbin, .swatchbin, .materialbin).

Layout (both FH2 1.0 and FH6 1.1, format reference: Nenkai/ForzaTools, MIT):
  header  'burG' u8 major u8 minor
          1.0: u16 blobCount, u32 headerSize, u32 totalSize                     (16 bytes)
          1.1: u16 0, u32 headerSize, u32 totalSize, u32 blobCount              (20 bytes)
  blob table, 24 bytes each: tag u32, u8 major, u8 minor, u16 metaCount, u32 metaOffset, u32 dataOffset,
          u32 dataSize, u32 dataSize2
  metadata blocks (one per blob, packed back to back): metaCount x (tag u32, u16 size<<4|version, u16 offset
          from this entry), then the metadata payloads. The whole block is padded to 4 (= headerSize).
  blob data, each padded to 4.
Tags are stored as little-endian u32, so 'Mesh' reads as b'hseM' on disk; this module uses the readable form.
"""
import struct


def _tag(raw):
    return raw[::-1].decode("latin1")


def _raw(tag):
    return tag.encode("latin1")[::-1]


class Meta:
    __slots__ = ("tag", "version", "data")

    def __init__(self, tag, version, data):
        self.tag, self.version, self.data = tag, version, bytes(data)

    def __repr__(self):
        return f"Meta({self.tag!r}, v{self.version}, {len(self.data)}B)"


class Blob:
    __slots__ = ("tag", "major", "minor", "meta", "data", "size2")

    def __init__(self, tag, major, minor, meta=None, data=b"", size2=None):
        self.tag, self.major, self.minor = tag, major, minor
        self.meta = list(meta or [])
        self.data = bytes(data)
        self.size2 = size2

    def get_meta(self, tag):
        for m in self.meta:
            if m.tag == tag:
                return m
        return None

    @property
    def name(self):
        m = self.get_meta("Name")
        return m.data.decode("latin1") if m else None

    @property
    def ident(self):
        m = self.get_meta("Id  ")
        return struct.unpack("<I", m.data)[0] if m else None

    def version_at_least(self, major, minor):
        return self.major > major or (self.major == major and self.minor >= minor)

    def __repr__(self):
        return f"Blob({self.tag!r} {self.major}.{self.minor} meta={self.meta} {len(self.data)}B)"


class Bundle:
    def __init__(self, major=1, minor=0, blobs=None, align=4):
        self.major, self.minor = major, minor
        self.blobs = list(blobs or [])
        # 4 everywhere except FH2 swatchbins, which pack the 0x53-byte header and the texture data without padding
        self.align = align

    # ---------------------------------------------------------------- read
    @classmethod
    def parse(cls, b):
        b = bytes(b)
        if b[:4] != b"burG":
            raise ValueError("not a Grub bundle")
        major, minor = b[4], b[5]
        if minor >= 1:
            _, header_size, total, count = struct.unpack_from("<hIII", b, 6)
            table = 20
        else:
            count, header_size, total = struct.unpack_from("<HII", b, 6)
            table = 16
        out = cls(major, minor, align=4 if header_size % 4 == 0 else 1)
        for i in range(count):
            raw, bmaj, bmin, mcount, moff, doff, dsize, dsize2 = struct.unpack_from("<4sBBHIIII", b, table + 24 * i)
            meta = []
            for j in range(mcount):
                e = moff + 8 * j
                mraw, flags, rel = struct.unpack_from("<4sHH", b, e)
                size = flags >> 4
                meta.append(Meta(_tag(mraw), flags & 0xF, b[e + rel:e + rel + size]))
            out.blobs.append(Blob(_tag(raw), bmaj, bmin, meta, b[doff:doff + dsize], dsize2))
        return out

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls.parse(f.read())

    # ---------------------------------------------------------------- write
    def serialize(self):
        table = 20 if self.minor >= 1 else 16
        meta_start = table + 24 * len(self.blobs)
        metabuf = bytearray()
        meta_offsets = []
        for blob in self.blobs:
            base = meta_start + len(metabuf)
            meta_offsets.append(base)
            heads = bytearray()
            payload = bytearray()
            n = len(blob.meta)
            for j, m in enumerate(blob.meta):
                entry = 8 * j
                rel = 8 * n + len(payload) - entry
                if len(m.data) >= 1 << 12 or rel > 0xFFFF:
                    raise ValueError(f"metadata {m.tag!r} too large")
                heads += struct.pack("<4sHH", _raw(m.tag), (len(m.data) << 4) | (m.version & 0xF), rel)
                payload += m.data
            metabuf += heads + payload
        header_size = meta_start + len(metabuf)
        header_size += (-header_size) % self.align
        data = bytearray()
        offsets = []
        for blob in self.blobs:
            offsets.append(header_size + len(data))
            data += blob.data
            data += b"\0" * ((-len(data)) % self.align)
        total = header_size + len(data)

        out = bytearray(b"burG" + bytes([self.major, self.minor]))
        if self.minor >= 1:
            out += struct.pack("<hIII", 0, header_size, total, len(self.blobs))
        else:
            out += struct.pack("<HII", len(self.blobs), header_size, total)
        for blob, moff, doff in zip(self.blobs, meta_offsets, offsets):
            size2 = len(blob.data) if blob.size2 is None else blob.size2
            out += struct.pack("<4sBBHIIII", _raw(blob.tag), blob.major, blob.minor, len(blob.meta), moff, doff,
                               len(blob.data), size2)
        out += metabuf
        out += b"\0" * (header_size - len(out))
        out += data
        assert len(out) == total
        return bytes(out)

    # ---------------------------------------------------------------- helpers
    def of(self, tag):
        return [b for b in self.blobs if b.tag == tag]

    def first(self, tag):
        for b in self.blobs:
            if b.tag == tag:
                return b
        return None
