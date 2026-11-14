"""Decode modelbin draw calls into plain numpy arrays (and back, see fh2_build.py).

Part = one Mesh blob: positions (N,3) float32 in model space (Y up, metres), normals (N,3), tangents (N,4),
uvs {semantic index: (N,2)}, joints (N,K) global skeleton bone indices, weights (N,K), tris (M,3) local indices.
"""
import numpy as np

import model
from grub import Bundle


# DXGI formats seen in the two games' vertex layouts
R32G32B32_FLOAT = 6
R16G16B16A16_FLOAT = 10
R16G16B16A16_SNORM = 13
R10G10B10A2_UNORM = 24
R8G8B8A8_UNORM = 28
R16G16_FLOAT = 34
R16G16_UNORM = 35
R16G16_SNORM = 37
FMT_SIZE = {R32G32B32_FLOAT: 12, R16G16B16A16_FLOAT: 8, R16G16B16A16_SNORM: 8, R10G10B10A2_UNORM: 4, R8G8B8A8_UNORM: 4,
            R16G16_FLOAT: 4, R16G16_UNORM: 4, R16G16_SNORM: 4}


def decode_element(raw, fmt):
    """raw: (N, size) uint8 -> (N, C) float32."""
    n = raw.shape[0]
    if fmt == R32G32B32_FLOAT:
        return raw.copy().view("<f4").reshape(n, 3)
    if fmt == R16G16B16A16_FLOAT:
        return raw.copy().view("<f2").reshape(n, 4).astype(np.float32)
    if fmt == R16G16_FLOAT:
        return raw.copy().view("<f2").reshape(n, 2).astype(np.float32)
    if fmt == R16G16B16A16_SNORM:
        return np.maximum(raw.copy().view("<i2").reshape(n, 4) / 32767.0, -1).astype(np.float32)
    if fmt == R16G16_SNORM:
        return np.maximum(raw.copy().view("<i2").reshape(n, 2) / 32767.0, -1).astype(np.float32)
    if fmt == R16G16_UNORM:
        return (raw.copy().view("<u2").reshape(n, 2) / 65535.0).astype(np.float32)
    if fmt == R8G8B8A8_UNORM:
        return (raw.reshape(n, 4) / 255.0).astype(np.float32)
    if fmt == R10G10B10A2_UNORM:
        v = raw.copy().view("<u4").reshape(n)
        return np.stack([(v & 1023) / 1023.0, ((v >> 10) & 1023) / 1023.0, ((v >> 20) & 1023) / 1023.0,
                         (v >> 30) / 3.0], axis=1).astype(np.float32)
    raise ValueError(f"vertex format {fmt}")


def oct_decode(e):
    """Octahedral unit vector decode, e in [-1,1]^2."""
    x, y = e[:, 0], e[:, 1]
    z = 1.0 - np.abs(x) - np.abs(y)
    t = np.clip(-z, 0, None)
    x = x + np.where(x >= 0, -t, t)
    y = y + np.where(y >= 0, -t, t)
    v = np.stack([x, y, z], axis=1)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class Part:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"Part({self.name!r} mat={self.material} lod={self.lod_mask:#x} v={len(self.positions)} t={len(self.tris)})"


class ModelData:
    """A parsed modelbin: skeleton, materials (MatI blobs by id), layouts, buffers, meshes."""

    def __init__(self, path_or_bytes):
        self.bundle = Bundle.load(path_or_bytes) if isinstance(path_or_bytes, str) else Bundle.parse(path_or_bytes)
        b = self.bundle
        self.skeleton = model.Skeleton.parse(b.first("Skel"))
        self.materials = {bl.ident: bl for bl in b.of("MatI")}
        self.layouts = [model.VertexLayout.parse(bl) for bl in b.of("VLay")]  # Mesh.vlay indexes this list
        self.vbufs = {bl.ident: model.Buffer.parse(bl) for bl in b.of("VerB")}
        self.skins = {bl.ident: model.Buffer.parse(bl) for bl in b.of("Skin")}
        ib = model.Buffer.parse(b.first("IndB"))
        self.indices = np.frombuffer(ib.data, dtype="<u2" if ib.stride == 2 else "<u4").astype(np.int64)
        self.meshes = [(bl.name, model.Mesh.parse(bl)) for bl in b.of("Mesh")]

    def _stream(self, buf_id):
        buf = self.vbufs[buf_id & 0xFFFFFFFF if buf_id < 0 else buf_id]
        return buf, np.frombuffer(buf.data, dtype=np.uint8).reshape(buf.count, buf.stride)

    def parts(self, include_shadow=False):
        out = []
        for name, m in self.meshes:
            if m.material < 0 and not include_shadow:
                continue
            idx = self.indices[m.face_start:m.face_start + m.index_count]
            layout = self.layouts[m.vlay]
            # every stream: vertex = index + base_vertex + offset/stride
            used = np.unique(idx)
            remap = np.full(used.max() + 1, -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            attrs = {}
            for s in m.streams:
                buf_id, slot, stride, offset = s[:4]
                buf, arr = self._stream(buf_id)
                rows = used + m.base_vertex + offset // stride
                data = arr[rows]
                pos = 0
                for el, packed in zip(layout.elements, layout.packed):
                    if el.slot != slot:
                        continue
                    size = FMT_SIZE[el.fmt]
                    attrs[(layout.names[el.semantic], el.index)] = decode_element(np.ascontiguousarray(data[:, pos:pos + size]), el.fmt)
                    pos += size
                if pos != stride:
                    raise ValueError(f"{name}: slot {slot} decodes {pos} of {stride} bytes")
            p = attrs[("POSITION", 0)]
            uvs = {}
            for (sem, i), v in attrs.items():
                if sem == "TEXCOORD":
                    ou, su, ov, sv = m.uv[4 * i:4 * i + 4]
                    uvs[i] = np.stack([v[:, 0] * su + ou, v[:, 1] * sv + ov], axis=1)
            tris = remap[idx].reshape(-1, 3)
            if m.fh6:
                # POSITION is R16G16B16A16_SNORM scaled by the mesh; its 4th lane is the normal's x, NORMAL (R16G16_SNORM)
                # holds y and z. TANGENT is R10G10B10A2_UNORM xyz*2-1 (unnormalized, direction = UV0 tangent); the
                # handedness comes from the UVs.
                normals = np.stack([p[:, 3], attrs[("NORMAL", 0)][:, 0], attrs[("NORMAL", 0)][:, 1]], axis=1)
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                p = p[:, :3] * np.array(m.pos_scale[:3], np.float32) + np.array(m.pos_offset[:3], np.float32)
                t = attrs.get(("TANGENT", 0))
                tangents = None
                if t is not None:
                    tv = oct_or_unorm(t[:, :3])
                    tangents = np.concatenate([tv, uv_handedness(p, normals, tv, uvs.get(0), tris)[:, None]], axis=1)
            else:
                normals = attrs[("NORMAL", 0)][:, :3]
                t = attrs.get(("TANGENT", 0))
                tangents = t if t is not None else None
            joints, weights = self._skin(m, used)
            out.append(Part(name=name, mesh=m, material=m.material, lod_mask=m.lod_mask, flags=m.flags,
                            positions=p.astype(np.float32), normals=normals.astype(np.float32), tangents=tangents,
                            uvs=uvs, color=attrs.get(("COLOR", 0)), joints=joints, weights=weights, tris=tris))
        return out

    def _skin(self, m, used):
        attr_stream = [s for s in m.streams if s[1] != 0]
        skin = self.skins.get(m.skin)
        if not attr_stream or skin is None:
            return None, None
        buf_id, slot, stride, offset = attr_stream[0][:4]
        rows = used + m.base_vertex + offset // stride
        arr = np.frombuffer(skin.data, dtype="<f2").reshape(skin.count, skin.elements, 2)[rows].astype(np.float32)
        weights, joints = arr[:, :, 0], np.rint(arr[:, :, 1]).astype(np.int64)
        return joints, weights


def uv_tangents(p, uv, tris):
    """Per-vertex (tangent, bitangent) accumulated from triangle UV gradients."""
    dp1, dp2 = p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]]
    du1, du2 = uv[tris[:, 1]] - uv[tris[:, 0]], uv[tris[:, 2]] - uv[tris[:, 0]]
    r = du1[:, 0] * du2[:, 1] - du2[:, 0] * du1[:, 1]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    ft = (dp1 * du2[:, 1:2] - dp2 * du1[:, 1:2]) / r[:, None]
    fb = (dp2 * du1[:, 0:1] - dp1 * du2[:, 0:1]) / r[:, None]
    t, b = np.zeros_like(p), np.zeros_like(p)
    for k in range(3):
        np.add.at(t, tris[:, k], ft)
        np.add.at(b, tris[:, k], fb)
    return t, b


def uv_handedness(p, n, t, uv, tris):
    if uv is None:
        return np.ones(len(p), np.float32)
    _, b = uv_tangents(p.astype(np.float64), uv.astype(np.float64), tris)
    return np.where(np.sum(np.cross(n, t) * b, axis=1) < 0, -1.0, 1.0).astype(np.float32)


def oct_or_unorm(xyz01):
    """FH6 TANGENT is R10G10B10A2_UNORM; xyz in [0,1] -> [-1,1], normalized."""
    v = xyz01 * 2.0 - 1.0
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def world_matrices(skeleton):
    """Bind pose world matrices (row-vector convention: world = local @ parent_world)."""
    mats = []
    for b in skeleton.bones:
        local = np.array(b.matrix, dtype=np.float64).reshape(4, 4)
        mats.append(local if b.parent < 0 else local @ mats[b.parent])
    return np.array(mats)
