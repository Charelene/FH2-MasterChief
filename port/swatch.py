"""Textures: FH6 PC swatchbins in, FH2 Durango (Xbox One) swatchbins out.

FH6 (TXCB 0.x, "PC"): linear BCn mip chain; header layout per Nenkai/ForzaTools PCTextureContentHeader.
FH2 (TXCB 2.0, Durango): TXCH metadata (0x23 bytes) = guid, u16 width, height, depth, full width, full height,
u16 unk (1), u8 mips, u8 full mips, u8 base mip, u32 flags (tile mode bits 0-4, encoding 5-10, transcoding 11-16,
bits 17-19 unk, colour profile 20-22, top nibble unk), then the mip chain tiled for the GPU (tile mode 14 =
XG_TILE_MODE_2D_THIN in every FH2 driver texture). Tiling uses Microsoft's xg.dll from the local GDK (never shipped).
"""
import ctypes
import os
import struct
import subprocess
import tempfile
import uuid

import numpy as np

from grub import Blob, Bundle, Meta

XG_DLL = os.environ.get("XG_DLL", r"S:\Forza\Projects\FH2MP\scratch\gdk\gdk-2020\Microsoft GXDK x86-x86_en-us\Microsoft GDK\200600\GXDK\bin\XboxOne\xg.dll")
TEXCONV = os.environ.get("TEXCONV", r"S:\DevCache\tools\texconv\texconv.exe")

# DXGI
BC1_UNORM, BC1_SRGB, BC3_UNORM, BC3_SRGB, BC4_UNORM, BC5_UNORM, BC5_SNORM = 71, 72, 77, 78, 80, 83, 84
BC7_UNORM, BC7_SRGB = 98, 99
R8G8B8A8_UNORM = 28
ENCODING = [(71, 72), (74, 75), (77, 78), (80, 80), (81, 81), (83, 83), (84, 84), (95, 0), (96, 0), (98, 99), (2, 0),
            (11, 0), (10, 0), (28, 29), (85, 0), (86, 0), (0, 0), (0, 0), (0, 0), (61, 0), (65, 0)]
BLOCK_BYTES = {71: 8, 72: 8, 80: 8, 81: 8, 74: 16, 75: 16, 77: 16, 78: 16, 83: 16, 84: 16, 95: 16, 96: 16, 98: 16, 99: 16}
TILE_2D_THIN = 14


def mip_dims(w, h, mips):
    out = []
    for _ in range(mips):
        out.append((w, h))
        w, h = max(1, w // 2), max(1, h // 2)
    return out


def linear_mip_size(fmt, w, h):
    if fmt in BLOCK_BYTES:
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * BLOCK_BYTES[fmt]
    if fmt in (28, 29):
        return w * h * 4
    raise ValueError(f"format {fmt}")


def row_pitch(fmt, w):
    return max(1, (w + 3) // 4) * BLOCK_BYTES[fmt] if fmt in BLOCK_BYTES else w * 4


# ------------------------------------------------------------------------------------------------ FH6 (PC) read
class Texture:
    """A linear mip chain: fmt (DXGI), width, height, mips [bytes per mip]."""

    def __init__(self, fmt, width, height, mips):
        self.fmt, self.width, self.height, self.mips = fmt, width, height, mips


def read_fh6(path):
    b = Bundle.load(path)
    blob = b.first("TXCB")
    h = blob.get_meta("TXCH").data
    off1, off2 = struct.unpack_from("<II", h, 0)
    w, hgt, depth, slices, mips, unkbits, transcoding, cp2, cp = struct.unpack_from("<iiIHBBIII", h, 0x18)
    params = []
    pos = off1
    while True:
        vo = struct.unpack_from("<i", h, pos)[0]
        if vo == -1:
            break
        nxt = struct.unpack_from("<i", h, pos + 4)[0]
        params.append(struct.unpack_from("<i", h, vo)[0])
        if nxt == -1:
            break
        pos = nxt
    if transcoding > 1:
        raise ValueError(f"{path}: transcoded texture ({transcoding})")
    lin, srgb = ENCODING[params[0]]
    fmt = srgb if cp else lin
    data = blob.data
    out, o = [], 0
    for mw, mh in mip_dims(w, hgt, mips):
        n = linear_mip_size(fmt, mw, mh)
        out.append(data[o:o + n])
        o += n
    if o != len(data):
        raise ValueError(f"{path}: {len(data) - o} bytes after the mip chain")
    return Texture(fmt, w, hgt, out)


# ------------------------------------------------------------------------------------------------ DDS
DDS_FLAGS = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000 | 0x20000


def write_dds(path, tex):
    w, h = tex.width, tex.height
    hdr = struct.pack("<4sIIIIIII44x", b"DDS ", 124, DDS_FLAGS, h, w, linear_mip_size(tex.fmt, w, h), 0, len(tex.mips))
    pf = struct.pack("<II4s20x", 32, 0x4, b"DX10")
    caps = struct.pack("<IIII4x", 0x1000 | 0x400000 | 0x8, 0, 0, 0)
    dx10 = struct.pack("<IIIII", tex.fmt, 3, 0, 1, 0)
    with open(path, "wb") as f:
        f.write(hdr + pf + caps + dx10 + b"".join(tex.mips))


def read_dds(path):
    b = open(path, "rb").read()
    if b[:4] != b"DDS ":
        raise ValueError("not a DDS")
    h, w = struct.unpack_from("<II", b, 12)
    mips = max(1, struct.unpack_from("<I", b, 28)[0])
    fourcc = b[84:88]
    if fourcc == b"DX10":
        fmt = struct.unpack_from("<I", b, 128)[0]
        o = 148
    else:
        raise ValueError(f"DDS without DX10 header ({fourcc!r})")
    out = []
    for mw, mh in mip_dims(w, h, mips):
        n = linear_mip_size(fmt, mw, mh)
        out.append(b[o:o + n])
        o += n
    return Texture(fmt, w, h, out)


def texconv(src, outdir, *args):
    r = subprocess.run([TEXCONV, "-nologo", "-y", "-dx10", "-o", outdir, *args, src], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"texconv failed: {r.stdout}{r.stderr}")
    return os.path.join(outdir, os.path.splitext(os.path.basename(src))[0] + ".dds")


SRGB_FORMATS = (29, 72, 75, 78, 99)


def to_rgba(tex):
    """Decode mip 0 of a BCn texture to (H, W, 4) uint8. Values stay as stored: an sRGB texture decodes to
    R8G8B8A8_UNORM_SRGB (sRGB-encoded bytes); texconv would otherwise linearize it."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.dds")
        one = Texture(tex.fmt, tex.width, tex.height, tex.mips[:1])
        write_dds(src, one)
        os.makedirs(os.path.join(d, "out"))
        out = texconv(src, os.path.join(d, "out"), "-f", "R8G8B8A8_UNORM_SRGB" if tex.fmt in SRGB_FORMATS else "R8G8B8A8_UNORM", "-m", "1")
        t = read_dds(out)
    return np.frombuffer(t.mips[0], np.uint8).reshape(t.height, t.width, 4).copy()


def from_rgba(img, fmt, mips=0, srgb_in=None):
    """Encode (H, W, 4) uint8 to a BCn mip chain with texconv. For an sRGB target the bytes are taken as sRGB-encoded
    (input DDS tagged R8G8B8A8_UNORM_SRGB, so texconv converts nothing); mips are filtered by texconv."""
    h, w = img.shape[:2]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.dds")
        in_fmt = 29 if fmt in SRGB_FORMATS else R8G8B8A8_UNORM
        write_dds(src, Texture(in_fmt, w, h, [np.ascontiguousarray(img, np.uint8).tobytes()]))
        os.makedirs(os.path.join(d, "out"))
        names = {71: "BC1_UNORM", 72: "BC1_UNORM_SRGB", 77: "BC3_UNORM", 78: "BC3_UNORM_SRGB", 80: "BC4_UNORM",
                 83: "BC5_UNORM", 98: "BC7_UNORM", 99: "BC7_UNORM_SRGB"}
        out = texconv(src, os.path.join(d, "out"), "-f", names[fmt], "-m", str(mips))
        return read_dds(out)


# ------------------------------------------------------------------------------------------------ XG (Durango tiling)
class XG_SAMPLE_DESC(ctypes.Structure):
    _fields_ = [("Count", ctypes.c_uint), ("Quality", ctypes.c_uint)]


class XG_TEXTURE2D_DESC(ctypes.Structure):
    _fields_ = [("Width", ctypes.c_uint), ("Height", ctypes.c_uint), ("MipLevels", ctypes.c_uint), ("ArraySize", ctypes.c_uint),
                ("Format", ctypes.c_int), ("SampleDesc", XG_SAMPLE_DESC), ("Usage", ctypes.c_int), ("BindFlags", ctypes.c_uint),
                ("CPUAccessFlags", ctypes.c_uint), ("MiscFlags", ctypes.c_uint), ("ESRAMOffsetBytes", ctypes.c_uint),
                ("ESRAMUsageBytes", ctypes.c_uint), ("TileMode", ctypes.c_int), ("Pitch", ctypes.c_uint)]


class XG_MIPLEVEL_LAYOUT(ctypes.Structure):
    _fields_ = [("SizeBytes", ctypes.c_uint64), ("OffsetBytes", ctypes.c_uint64), ("Slice2DSizeBytes", ctypes.c_uint64),
                ("PitchPixels", ctypes.c_uint32), ("PitchBytes", ctypes.c_uint32), ("AlignmentBytes", ctypes.c_uint32),
                ("PaddedWidthElements", ctypes.c_uint32), ("PaddedHeightElements", ctypes.c_uint32),
                ("PaddedDepthOrArraySize", ctypes.c_uint32), ("WidthElements", ctypes.c_uint32), ("HeightElements", ctypes.c_uint32),
                ("DepthOrArraySize", ctypes.c_uint32), ("SampleCount", ctypes.c_uint32), ("TileMode", ctypes.c_int),
                ("BankRotationAddressBitMask", ctypes.c_uint64), ("BankRotationBytesPerSlice", ctypes.c_uint64),
                ("SliceDepthElements", ctypes.c_uint32)]


class XG_PLANE_LAYOUT(ctypes.Structure):
    _fields_ = [("Usage", ctypes.c_int), ("SizeBytes", ctypes.c_uint64), ("BaseOffsetBytes", ctypes.c_uint64),
                ("BaseAlignmentBytes", ctypes.c_uint64), ("BytesPerElement", ctypes.c_uint32), ("MipLayout", XG_MIPLEVEL_LAYOUT * 15)]


class XG_RESOURCE_LAYOUT(ctypes.Structure):
    _fields_ = [("SizeBytes", ctypes.c_uint64), ("BaseAlignmentBytes", ctypes.c_uint64), ("MipLevels", ctypes.c_uint32),
                ("Planes", ctypes.c_uint32), ("Plane", XG_PLANE_LAYOUT * 4), ("Dimension", ctypes.c_int)]


_xg = None


def _lib():
    global _xg
    if _xg is None:
        _xg = ctypes.CDLL(XG_DLL)
        _xg.XGCreateTexture2DComputer.argtypes = [ctypes.POINTER(XG_TEXTURE2D_DESC), ctypes.POINTER(ctypes.c_void_p)]
        _xg.XGCreateTexture2DComputer.restype = ctypes.c_long
    return _xg


class Computer:
    """XGTextureAddressComputer vtable: 0 AddRef, 1 Release, 2 GetResourceLayout, 3 GetResourceSizeBytes,
    4 GetResourceBaseAlignmentBytes, 5 GetMipLevelOffsetBytes, 6 GetTexelElementOffsetBytes, 7 GetTexelCoordinate,
    8 CopyIntoSubresource, 9 CopyFromSubresource, ..."""

    def __init__(self, fmt, w, h, mips, tile=TILE_2D_THIN):
        d = XG_TEXTURE2D_DESC(Width=w, Height=h, MipLevels=mips, ArraySize=1, Format=fmt, SampleDesc=XG_SAMPLE_DESC(1, 0),
                              Usage=0, BindFlags=0x8, TileMode=tile)
        self.ptr = ctypes.c_void_p()
        hr = _lib().XGCreateTexture2DComputer(ctypes.byref(d), ctypes.byref(self.ptr))
        if hr < 0:
            raise OSError(f"XGCreateTexture2DComputer 0x{hr & 0xFFFFFFFF:08x}")
        vt = ctypes.cast(ctypes.cast(self.ptr, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
        P = ctypes.c_void_p
        self._layout = ctypes.CFUNCTYPE(ctypes.c_long, P, ctypes.POINTER(XG_RESOURCE_LAYOUT))(vt[2])
        self._into = ctypes.CFUNCTYPE(ctypes.c_long, P, P, ctypes.c_uint32, ctypes.c_uint32, P, ctypes.c_uint32, ctypes.c_uint32)(vt[8])
        self._from = ctypes.CFUNCTYPE(ctypes.c_long, P, P, ctypes.c_uint32, ctypes.c_uint32, P, ctypes.c_uint32, ctypes.c_uint32)(vt[9])
        self._release = ctypes.CFUNCTYPE(ctypes.c_ulong, P)(vt[1])
        self.layout = XG_RESOURCE_LAYOUT()
        hr = self._layout(self.ptr, ctypes.byref(self.layout))
        if hr < 0:
            raise OSError(f"GetResourceLayout 0x{hr & 0xFFFFFFFF:08x}")
        self.fmt, self.w, self.h, self.mips = fmt, w, h, mips

    def close(self):
        if self.ptr:
            self._release(self.ptr)
            self.ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    @property
    def size(self):
        return self.layout.SizeBytes

    def detile(self, tiled):
        buf = ctypes.create_string_buffer(bytes(tiled), max(len(tiled), self.size))
        out = []
        for m, (mw, mh) in enumerate(mip_dims(self.w, self.h, self.mips)):
            n = linear_mip_size(self.fmt, mw, mh)
            dst = ctypes.create_string_buffer(n)
            hr = self._from(self.ptr, ctypes.cast(dst, ctypes.c_void_p), 0, m, ctypes.cast(buf, ctypes.c_void_p), row_pitch(self.fmt, mw), 0)
            if hr < 0:
                raise OSError(f"CopyFromSubresource mip {m}: 0x{hr & 0xFFFFFFFF:08x}")
            out.append(dst.raw)
        return out

    def tile(self, mips):
        buf = ctypes.create_string_buffer(self.size)
        for m, ((mw, mh), data) in enumerate(zip(mip_dims(self.w, self.h, self.mips), mips)):
            src = ctypes.create_string_buffer(bytes(data), len(data))
            hr = self._into(self.ptr, ctypes.cast(buf, ctypes.c_void_p), 0, m, ctypes.cast(src, ctypes.c_void_p), row_pitch(self.fmt, mw), 0)
            if hr < 0:
                raise OSError(f"CopyIntoSubresource mip {m}: 0x{hr & 0xFFFFFFFF:08x}")
        return buf.raw


# ------------------------------------------------------------------------------------------------ FH2 (Durango)
class DurangoHeader:
    FMT = "<16sHHHHHHBBBI"

    def __init__(self, data):
        (self.guid, self.width, self.height, self.depth, self.full_width, self.full_height, self.unk, self.mips,
         self.full_mips, self.base_mip, self.flags) = struct.unpack(self.FMT, data)

    def pack(self):
        return struct.pack(self.FMT, self.guid, self.width, self.height, self.depth, self.full_width, self.full_height, self.unk,
                           self.mips, self.full_mips, self.base_mip, self.flags)

    @property
    def tile_mode(self):
        return self.flags & 0x1F

    @property
    def fmt(self):
        enc = (self.flags >> 5) & 0x3F
        cp = (self.flags >> 20) & 0x7
        lin, srgb = ENCODING[enc]
        return srgb if cp and srgb else lin


def read_fh2(path):
    b = Bundle.load(path)
    blob = b.first("TXCB")
    hdr = DurangoHeader(blob.get_meta("TXCH").data)
    with Computer(hdr.fmt, hdr.width, hdr.height, hdr.mips, hdr.tile_mode) as c:
        if c.size != len(blob.data):
            raise ValueError(f"{path}: XG size {c.size} != data {len(blob.data)}")
        mips = c.detile(blob.data)
    return hdr, Texture(hdr.fmt, hdr.width, hdr.height, mips)


def write_fh2(path, tex, flags_from, guid=None):
    """Write tex (linear mip chain) as an FH2 swatchbin. flags_from: a DurangoHeader of an FH2 texture with the same
    format, whose flag bits (tile mode, encoding, colour profile, unknown bits) are kept."""
    hdr = DurangoHeader(flags_from.pack())
    if hdr.fmt != tex.fmt:
        raise ValueError(f"format {tex.fmt} != template {hdr.fmt}")
    hdr.guid = (guid or uuid.uuid4()).bytes_le
    hdr.width = hdr.full_width = tex.width
    hdr.height = hdr.full_height = tex.height
    hdr.mips = hdr.full_mips = len(tex.mips)
    hdr.base_mip = 0
    with Computer(tex.fmt, tex.width, tex.height, len(tex.mips), hdr.tile_mode) as c:
        data = c.tile(tex.mips)
    b = Bundle(1, 0, [Blob("TXCB", 2, 0, [Meta("TXCH", 0, hdr.pack())], data)], align=1)
    raw = b.serialize()
    if path:
        with open(path, "wb") as f:
            f.write(raw)
    return raw
