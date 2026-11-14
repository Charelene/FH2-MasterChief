"""Blob payloads of ForzaTech .modelbin files, FH2 (1.0.0.64) and FH6.

Each parser keeps every field it reads so that serialize() rebuilds the blob byte for byte (tests/test_model.py).
Field names follow Nenkai/ForzaTools (MIT) where its layout matched; unknowns keep its Unk names.
"""
import struct

from grub import Blob, Bundle


class Reader:
    def __init__(self, data, pos=0):
        self.d, self.o = data, pos

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.d, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    def bytes(self, n):
        v = self.d[self.o:self.o + n]
        self.o += n
        return v

    def string(self):
        n = self.take("I")
        return self.bytes(n).decode("latin1")

    def done(self):
        return self.o == len(self.d)


def pack(fmt, *v):
    return struct.pack("<" + fmt, *v)


# ------------------------------------------------------------------------------------------------ Skel
class Bone:
    __slots__ = ("name", "parent", "first_child", "next_sibling", "matrix")

    def __init__(self, name, parent, first_child, next_sibling, matrix):
        self.name, self.parent, self.first_child, self.next_sibling = name, parent, first_child, next_sibling
        self.matrix = tuple(matrix)  # 4x4 row-major, local to parent, translation in the last row (DirectX)

    def __repr__(self):
        return f"Bone({self.name!r}, parent={self.parent})"


class Skeleton:
    """Skel 0.0 (FH2): u16 count, bones; Skel 1.0 (FH6): the same plus a trailing u32."""

    def __init__(self, bones, tail=b""):
        self.bones, self.tail = bones, tail

    @classmethod
    def parse(cls, blob):
        r = Reader(blob.data)
        bones = []
        for _ in range(r.take("H")):
            name = r.string()
            p, fc, nx = r.take("hhh")
            bones.append(Bone(name, p, fc, nx, r.take("16f")))
        return cls(bones, r.d[r.o:])

    def serialize(self):
        out = bytearray(pack("H", len(self.bones)))
        for b in self.bones:
            nm = b.name.encode("latin1")
            out += pack("I", len(nm)) + nm + pack("hhh", b.parent, b.first_child, b.next_sibling) + pack("16f", *b.matrix)
        return bytes(out + self.tail)

    def index(self, name):
        for i, b in enumerate(self.bones):
            if b.name == name:
                return i
        return -1


# ------------------------------------------------------------------------------------------------ buffers
class Buffer:
    """VerB / IndB / Skin payload: u32 count, u32 totalSize, u16 stride, u8 elements, u8 pad, i32 dxgi, data."""

    def __init__(self, count, stride, elements, dxgi, data, pad=0, total=None):
        self.count, self.stride, self.elements, self.dxgi, self.data, self.pad = count, stride, elements, dxgi, data, pad
        self.total = total

    @classmethod
    def parse(cls, blob):
        r = Reader(blob.data)
        count, total, stride, elements, pad, dxgi = r.take("IIHBBi")
        data = r.bytes(count * stride)
        if not r.done() or total != count * stride:
            raise ValueError(f"{blob.tag}: unexpected buffer layout")
        return cls(count, stride, elements, dxgi, data, pad, total)

    def serialize(self):
        total = self.count * self.stride if self.total is None else self.total
        return pack("IIHBBi", self.count, total, self.stride, self.elements, self.pad, self.dxgi) + self.data


# ------------------------------------------------------------------------------------------------ VLay
class Element:
    __slots__ = ("semantic", "index", "slot", "fmt", "offset", "step")

    def __init__(self, semantic, index, slot, fmt, offset, step):
        self.semantic, self.index, self.slot, self.fmt, self.offset, self.step = semantic, index, slot, fmt, offset, step

    def __repr__(self):
        return f"El(sem{self.semantic}.{self.index} slot{self.slot} fmt{self.fmt} off{self.offset})"


class VertexLayout:
    def __init__(self, names, elements, packed, flags):
        self.names, self.elements, self.packed, self.flags = names, elements, packed, flags

    @classmethod
    def parse(cls, blob):
        r = Reader(blob.data)
        names = [r.string() for _ in range(r.take("H"))]
        n = r.take("H")
        els = [Element(*r.take("hhiiii")) for _ in range(n)]
        packed = [r.take("i") for _ in range(n)]
        flags = r.take("I") if blob.version_at_least(1, 1) else None
        if not r.done():
            raise ValueError("VLay: trailing bytes")
        return cls(names, els, packed, flags)

    def serialize(self):
        out = bytearray(pack("H", len(self.names)))
        for nm in self.names:
            out += pack("I", len(nm)) + nm.encode("latin1")
        out += pack("H", len(self.elements))
        for e in self.elements:
            out += pack("hhiiii", e.semantic, e.index, e.slot, e.fmt, e.offset, e.step)
        for p in self.packed:
            out += pack("i", p)
        if self.flags is not None:
            out += pack("I", self.flags)
        return bytes(out)

    def describe(self):
        return [(self.names[e.semantic], e.index, e.slot, e.fmt, p) for e, p in zip(self.elements, self.packed)]


# ------------------------------------------------------------------------------------------------ Mesh
class Mesh:
    """One draw call. Two layouts, worked out from FH2 stig/default and the FH6 armour (all blobs round-trip):

    FH2 1.7:  h material, h unk2, H lod_mask (bit0 = LODS, bit1+n = LODn), b lod_min, b lod_max, B flags (0x11 opaque,
              0x1c glass visor, 0x09 shadow), B b9, B b10, B influences (1/2/4 = Skin pairs per vertex), 13s pad
    FH6 1.12: h a0 (-1), h material, i a4 (-1), H a8 (0xc2/0xc3), H lod_mask, b lod_min, b lod_max, B flags (0x19),
              B b9, B b10, B influences, 16s pad
    both:     I face_start, i base_vertex, I index_count, I tri_count, f unk_f, I num_verts
    FH6 only: I n, n x I (2 per vertex, unused here)
    both:     I vlay (index of the VLay blob, in blob order), i n, n x streams (FH2: i buffer_id, i slot, I stride,
              I offset; FH6 adds I extra), i morph (-1), i skin (Skin blob id; FH2 VerB/Skin ids pair up), i n, n x i bones, I v1, 20 f uv transforms (per TEXCOORD index:
              offset_u, scale_u, offset_v, scale_v)
    FH6 only: 4 f position scale, 4 f position offset (POSITION is R16G16B16A16_SNORM)
    Positions: buffer -1 (FH2, shared R32G32B32_FLOAT) or the mesh's position stream; attribute vertex =
    index + base_vertex.
    """

    def __init__(self, fh6, **f):
        self.fh6 = fh6
        self.__dict__.update(f)

    @classmethod
    def parse(cls, blob):
        fh6 = blob.version_at_least(1, 10)
        r = Reader(blob.data)
        f = {}
        if fh6:
            f["a0"], f["material"], f["a4"], f["a8"] = r.take("hhiH")
        else:
            f["material"], f["unk2"] = r.take("hh")
        f["lod_mask"], f["lod_min"], f["lod_max"], f["flags"], f["b9"], f["b10"], f["influences"] = r.take("HbbBBBB")
        f["pad"] = r.bytes(16 if fh6 else 13)
        (f["face_start"], f["base_vertex"], f["index_count"], f["tri_count"], f["unk_f"],
         f["num_verts"]) = r.take("IiIIfI")
        if fh6:
            f["vlist"] = list(r.take("%dI" % r.take("I")) or ())
        f["vlay"] = r.take("I")
        n = r.take("i")
        f["streams"] = [r.take("iiIII" if fh6 else "iiII") for _ in range(n)]
        f["morph"], f["skin"] = r.take("ii")
        n = r.take("i")
        f["bones"] = list(r.take("%di" % n)) if n else []
        f["v1"] = r.take("I")
        f["uv"] = r.take("20f")
        if fh6:
            f["pos_scale"], f["pos_offset"] = r.take("4f"), r.take("4f")
        if not r.done():
            raise ValueError(f"Mesh {blob.major}.{blob.minor}: {len(r.d) - r.o} bytes left")
        return cls(fh6, **f)

    def serialize(self):
        out = bytearray()
        if self.fh6:
            out += pack("hhiH", self.a0, self.material, self.a4, self.a8)
        else:
            out += pack("hh", self.material, self.unk2)
        out += pack("HbbBBBB", self.lod_mask, self.lod_min, self.lod_max, self.flags, self.b9, self.b10, self.influences)
        out += self.pad
        out += pack("IiIIfI", self.face_start, self.base_vertex, self.index_count, self.tri_count, self.unk_f, self.num_verts)
        if self.fh6:
            out += pack("I", len(self.vlist)) + pack("%dI" % len(self.vlist), *self.vlist)
        out += pack("I", self.vlay) + pack("i", len(self.streams))
        for s in self.streams:
            out += pack("iiIII" if self.fh6 else "iiII", *s)
        out += pack("ii", self.morph, self.skin)
        out += pack("i", len(self.bones)) + pack("%di" % len(self.bones), *self.bones)
        out += pack("I", self.v1) + pack("20f", *self.uv)
        if self.fh6:
            out += pack("4f", *self.pos_scale) + pack("4f", *self.pos_offset)
        return bytes(out)


# ------------------------------------------------------------------------------------------------ Modl
class ModelHeader:
    def __init__(self, meshes, buffers, layouts, materials, rest):
        self.meshes, self.buffers, self.layouts, self.materials, self.rest = meshes, buffers, layouts, materials, rest

    @classmethod
    def parse(cls, blob):
        r = Reader(blob.data)
        return cls(*r.take("HHHH"), r.d[r.o:])

    def serialize(self):
        return pack("HHHH", self.meshes, self.buffers, self.layouts, self.materials) + self.rest


# ------------------------------------------------------------------------------------------------ MatI
def material_bundle(blob):
    """A MatI blob is a nested Grub bundle (material instance)."""
    return Bundle.parse(blob.data)


PARSERS = {
    "Skel": (Skeleton.parse, lambda o, b: o.serialize()),
    "VerB": (Buffer.parse, lambda o, b: o.serialize()),
    "IndB": (Buffer.parse, lambda o, b: o.serialize()),
    "Skin": (Buffer.parse, lambda o, b: o.serialize()),
    "VLay": (VertexLayout.parse, lambda o, b: o.serialize()),
    "Mesh": (Mesh.parse, lambda o, b: o.serialize()),
    "Modl": (ModelHeader.parse, lambda o, b: o.serialize()),
    "MatI": (material_bundle, lambda o, b: o.serialize()),
}
