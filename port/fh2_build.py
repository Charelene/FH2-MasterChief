"""Build the FH2 driver: MasterChief.zip = driver.modelbin (FH6 armour re-skinned onto the Stig skeleton) +
the Stig's driver.gr2/driver.gsf + Textures/*.swatchbin, and the Extra Options package around it.

python -I port/fh2_build.py            -> build/Master Chief/ (EO package) and build/preview/built.png
"""
import io
import json
import os
import shutil
import struct
import sys
import uuid
import zipfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I leaves the script dir out

import numpy as np  # noqa: E402

import fh6_source  # noqa: E402
import geometry  # noqa: E402
import model  # noqa: E402
import retarget  # noqa: E402
import swatch  # noqa: E402
from grub import Blob, Bundle, Meta  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "extracted")
BUILD = os.path.join(ROOT, "build")
DRIVER = "MasterChief"                     # DriverModels.MediaName and media/drivers/<DRIVER>.zip
TEX_DIR = "Textures"
TEX_GAME = "Game:\\Media\\drivers\\%s\\%s\\" % (DRIVER.lower(), TEX_DIR.lower())
STIG = os.path.join(EX, "fh2", "drivers", "stig")
SWATCH = os.path.join(EX, "fh2", "cartex", "driver", "swatches")
GAMEDB_SHA256 = "4d50013dc7705de05a074e4421679e16a3a1eda0816841861b0b7139ecf45517"  # FH2 1.0.0.64 media/stripped/gamedbRC.slt

# Stig helmet material (shader driversuit_stig) parameter name hashes, from its MatI
P_SPEC = 0xE39E03D9      # 0.015 suit, 0.5 helmet: reflection strength
P_FRESNEL = 0x3F741E30   # 0.015 suit, 0.06 helmet
P_GLOSS_SCALE = 0x35E3663D  # 1.0 helmet
P_UNK = 0xDED4DD3E       # 0.19 helmet
P_SPEC_COLOR = 0x0F873A00  # (0,0,0,1) suit, (0.97,0.96,0.99,1) helmet
T_GLOS, T_LCAO, T_DIFF, T_NRML = 0x01B54287, 0x0FEA383B, 0x6DD98CD9, 0x03AD47DD

# per FH6 material name: shader values (spec, fresnel, gloss scale, unk, spec colour); hard armour vs cloth vs visor
ARMOUR = (0.30, 0.05, 1.0, 0.19, (0.62, 0.66, 0.58, 1.0))
CLOTH = (0.04, 0.02, 1.0, 0.19, (0.25, 0.25, 0.25, 1.0))
VISOR = (0.65, 0.10, 1.0, 0.19, (1.0, 0.78, 0.35, 1.0))
LOOK = {"Mat_Techsuit": CLOTH, "Mat_Armor_Visor": VISOR, "Mat_Grey": CLOTH}
LCAO_LEVEL = 28          # the Stig helmet's lcao map is a flat ~28


# ------------------------------------------------------------------------------------------------ textures
def _template(prefix):
    for f in os.listdir(SWATCH):
        if f.startswith(prefix):
            return swatch.read_fh2(os.path.join(SWATCH, f))[0]
    raise FileNotFoundError(prefix)


def _guid(name):
    return uuid.uuid5(uuid.NAMESPACE_URL, "fh2-masterchief/" + name)


def _texname(stem):
    return "%s_%s.swatchbin" % (stem, _guid(stem))


def build_textures(md6, out):
    """-> {fh6 material id: {param hash: game path}}, plus {zip member: bytes}"""
    t_diff, t_glos, t_nrml, t_lcao = _template("stighelmet_diff"), _template("stighelmet_mask"), _template("stighelmet_nrml"), _template("stighelmet_lcao")
    files, params = {}, {}

    def put(stem, tex, tmpl):
        name = _texname(stem)
        files["%s/%s" % (TEX_DIR, name)] = swatch.write_fh2(None, tex, tmpl, _guid(stem))
        return TEX_GAME + name.lower()

    flat = np.full((64, 64, 4), LCAO_LEVEL, np.uint8)
    flat[..., 3] = 255
    lcao = put("mc_lcao", swatch.from_rgba(flat, swatch.BC4_UNORM), t_lcao)
    tex = fh6_source.material_textures(md6)
    for mid, blob in sorted(md6.materials.items()):
        name = blob.name
        stem = "mc_" + name.lower().replace("mat_armor_", "").replace("mat_", "")
        slots = tex.get(mid, {})
        if "bclr" in slots:
            base = fh6_source.texture_rgba(slots["bclr"]).astype(np.float32)
            rm = fh6_source.texture_rgba(slots["rough_metal"]).astype(np.float32) / 255.0 if "rough_metal" in slots else None
            if rm is not None and rm.shape[:2] != base.shape[:2]:
                rm = _resize(rm, base.shape[:2])
            if rm is not None:
                base[..., :3] *= (1.0 - 0.35 * rm[..., 1:2])               # metal reads darker without PBR
            if "emis" in slots:
                em = fh6_source.texture_rgba(slots["emis"]).astype(np.float32)
                base[..., :3] = np.maximum(base[..., :3], _resize(em, base.shape[:2])[..., :3])
            base[..., 3] = 255
            diff = swatch.from_rgba(np.clip(base, 0, 255).astype(np.uint8), swatch.BC7_SRGB)
            g = np.full(base.shape[:2], 120, np.float32) if rm is None else (1.0 - rm[..., 0]) * 255.0
            gl = np.zeros(base.shape[:2] + (4,), np.uint8)
            gl[..., 0] = gl[..., 1] = gl[..., 2] = np.clip(g, 0, 255).astype(np.uint8)
            gl[..., 3] = 255
            glos = swatch.from_rgba(gl, swatch.BC4_UNORM)
            nrml = swatch.read_fh6(slots["nrml"]) if "nrml" in slots else None
        else:  # Mat_Grey: the inside of the helmet
            c = np.zeros((64, 64, 4), np.uint8)
            c[...] = (45, 48, 45, 255)
            diff = swatch.from_rgba(c, swatch.BC7_SRGB)
            c[...] = (60, 60, 60, 255)
            glos = swatch.from_rgba(c, swatch.BC4_UNORM)
            nrml = None
        if nrml is None:
            n = np.zeros((64, 64, 4), np.uint8)
            n[...] = (128, 128, 0, 255)
            nrml = swatch.from_rgba(n, swatch.BC5_UNORM)
        params[mid] = {T_DIFF: put(stem + "_diff", diff, t_diff), T_GLOS: put(stem + "_glos", glos, t_glos),
                       T_NRML: put(stem + "_nrml", nrml, t_nrml), T_LCAO: lcao}
        print("textures", name, {k: os.path.basename(v)[:24] for k, v in params[mid].items()})
    return params, files


def _resize(img, hw):
    h, w = hw
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(w) * img.shape[1] / w).astype(int)
    return img[ys][:, xs]


# ------------------------------------------------------------------------------------------------ materials
def mtpr(values, textures):
    """MTPR 2.0: u8 count, entries (u16 2, u32 name hash, u8 type, value; type 0/1 float4, 2 float, 6 texture =
    u8 len + path + u32 crc32(lower path)), u32 crc32(value entries), u32 crc32(texture entries), u32 0."""
    vals, texs = b"", b""
    for h, v in values:
        if isinstance(v, tuple):
            vals += struct.pack("<HIB4f", 2, h, 1, *v)
        else:
            vals += struct.pack("<HIBf", 2, h, 2, v)
    for h, path in textures:
        p = path.encode("latin1")
        if len(p) > 255:
            raise ValueError("texture path too long: " + path)
        texs += struct.pack("<HIBB", 2, h, 6, len(p)) + p + struct.pack("<I", zlib.crc32(p.lower()))
    n = len(values) + len(textures)
    return bytes([n]) + vals + texs + struct.pack("<III", zlib.crc32(vals) if vals else 0, zlib.crc32(texs) if texs else 0, 0)


def material_blob(stig_helmet, mid, name, look, tex):
    inner = model.material_bundle(stig_helmet)
    spec, fres, gscale, unk, col = look
    values = [(P_GLOSS_SCALE, gscale), (P_UNK, unk), (P_SPEC, spec), (P_FRESNEL, fres), (P_SPEC_COLOR, col)]
    textures = [(T_GLOS, tex[T_GLOS]), (T_LCAO, tex[T_LCAO]), (T_DIFF, tex[T_DIFF]), (T_NRML, tex[T_NRML])]
    blobs = [b for b in inner.blobs if b.tag != "MTPR"] + [Blob("MTPR", 2, 0, [], mtpr(values, textures))]
    data = Bundle(inner.major, inner.minor, blobs, align=inner.align).serialize()
    return Blob("MatI", stig_helmet.major, stig_helmet.minor, [Meta("Name", 0, name.encode("latin1")), Meta("Id  ", 0, struct.pack("<I", mid))], data)


# ------------------------------------------------------------------------------------------------ geometry
def half(a):
    return np.asarray(a, np.float32).astype("<f2")


def build_model(md6, stig, tex_params):
    rt = retarget.Retarget(md6.skeleton, stig.skeleton)
    parts = md6.parts()
    lods = [(1, -1, -1)] + [(1 << (k + 1), k, k) for k in range(3)]  # LODS, LOD0..LOD2 (like the default player)
    verts = {"p": [], "n": [], "t": [], "uv": [], "j": [], "w": []}
    meshes = []  # (name, part index, lod tuple, shadow, index start, index count, vcount, bbox, material, unk_f, uvxf)
    indices = []
    vbase = 0
    istart = 0
    for k, part in enumerate(parts):
        p, n, t = rt.deform(part.positions, part.normals, part.tangents, part.joints, part.weights)
        j, w = rt.reweight(part.joints, part.weights, 4)
        uv = part.uvs[0].astype(np.float64)
        lo, hi = uv.min(0), uv.max(0)
        sc = np.where(hi - lo > 1e-6, hi - lo, 1.0)
        verts["p"].append(p)
        verts["n"].append(n)
        verts["t"].append(t)
        verts["uv"].append((uv - lo) / sc)
        verts["j"].append(j)
        verts["w"].append(w)
        tri = part.tris.astype(np.int64) + vbase
        indices.append(tri.ravel())
        bbox = np.concatenate([p.min(0), p.max(0)]).astype(np.float32)
        uvxf = (float(lo[0]), float(sc[0]), float(lo[1]), float(sc[1]))
        sets = [lods[0]] if part.lod_mask & 1 else lods[1:]
        for lod in sets:
            meshes.append(dict(name=part.name, lod=lod, shadow=False, start=istart, count=tri.size, verts=len(p), bbox=bbox,
                               material=part.material, unk_f=part.mesh.unk_f, uv=uvxf))
            if md6.materials[part.material].name != "Mat_Armor_Visor":
                meshes.append(dict(name="Shadow", lod=lod, shadow=True, start=istart, count=tri.size, verts=len(p), bbox=bbox,
                                   material=-1, unk_f=part.mesh.unk_f, uv=uvxf))
        vbase += len(p)
        istart += tri.size
    if vbase > 0xFFFF:
        raise ValueError(f"{vbase} vertices do not fit 16-bit indices")
    P = np.concatenate(verts["p"]).astype("<f4")
    N = np.concatenate(verts["n"])
    T = np.concatenate(verts["t"])
    UV = np.concatenate(verts["uv"])
    J = np.concatenate(verts["j"])
    W = np.concatenate(verts["w"])
    I = np.concatenate(indices).astype("<u2")
    nv = len(P)

    # attribute stream = Stig VLay #1: NORMAL f16x4, TEXCOORD0 u16x2, TEXCOORD2 u16x2, TANGENT f16x4, COLOR rgba8
    attr = np.zeros((nv, 28), np.uint8)
    attr[:, 0:8] = half(np.concatenate([N, np.zeros((nv, 1))], 1)).view(np.uint8).reshape(nv, 8)
    uv16 = np.clip(np.rint(UV * 65535), 0, 65535).astype("<u2")
    attr[:, 8:12] = uv16.view(np.uint8).reshape(nv, 4)
    attr[:, 12:16] = uv16.view(np.uint8).reshape(nv, 4)
    attr[:, 16:24] = half(T).view(np.uint8).reshape(nv, 8)
    attr[:, 24:28] = 255
    skin = np.zeros((nv, 4, 2), np.float32)
    skin[:, :, 0], skin[:, :, 1] = W, J
    skin = half(skin).view(np.uint8).reshape(nv, 16)

    stig_vlay = stig.bundle.of("VLay")
    vl_main = [b for b in stig_vlay if b.ident == 1][0]
    vl_pos = [b for b in stig_vlay if b.ident == 0xFFFFFFFF][0]
    blobs = [stig.bundle.first("Skel")]
    stig_helmet = stig.materials[5]
    for mid, blob in sorted(md6.materials.items()):
        blobs.append(material_blob(stig_helmet, mid, blob.name, LOOK.get(blob.name, ARMOUR), tex_params[mid]))
    stig_mesh = model.Mesh.parse(stig.bundle.of("Mesh")[0])
    lod_union = 0
    for m in meshes:
        mask, lmin, lmax = m["lod"]
        lod_union |= mask
        uvs = list(m["uv"]) + [0.0, 1.0, 0.0, 1.0] + list(m["uv"]) + [0.0, 1.0, 0.0, 1.0] * 2
        mesh = model.Mesh(False, material=m["material"], unk2=0, lod_mask=mask, lod_min=lmin, lod_max=lmax,
                          flags=0x09 if m["shadow"] else 0x11, b9=0, b10=0, influences=4, pad=stig_mesh.pad,
                          face_start=m["start"], base_vertex=0, index_count=m["count"], tri_count=m["count"] // 3,
                          unk_f=m["unk_f"], num_verts=0 if m["shadow"] else m["verts"], vlay=1 if m["shadow"] else 0,
                          streams=[(-1, 0, 12, 0)] if m["shadow"] else [(-1, 0, 12, 0), (0, 1, 28, 0)],
                          morph=-1, skin=0, bones=[], v1=0, uv=tuple(uvs))
        blobs.append(Blob("Mesh", 1, 7, [Meta("Name", 0, m["name"].encode()), Meta("BBox", 0, m["bbox"].tobytes())], mesh.serialize()))
    ident = lambda i: [Meta("Id  ", 0, struct.pack("<I", i & 0xFFFFFFFF))]  # noqa: E731
    blobs.append(Blob("IndB", 1, 0, ident(0), model.Buffer(len(I), 2, 1, 57, I.tobytes()).serialize()))
    blobs.append(Blob("VLay", vl_main.major, vl_main.minor, ident(0), vl_main.data))
    blobs.append(Blob("VLay", vl_pos.major, vl_pos.minor, ident(-1), vl_pos.data))
    blobs.append(Blob("VerB", 1, 0, ident(-1), model.Buffer(nv, 12, 1, 6, P.tobytes()).serialize()))
    blobs.append(Blob("VerB", 1, 0, ident(0), model.Buffer(nv, 28, 5, 10, attr.tobytes()).serialize()))
    blobs.append(Blob("Skin", 1, 0, ident(0), model.Buffer(nv, 16, 4, 34, skin.tobytes()).serialize()))
    bbox = np.concatenate([P.min(0), P.max(0)]).astype("<f4")
    refs = []
    for t in tex_params.values():
        for path in t.values():
            c = zlib.crc32(path.lower().encode("latin1"))
            if c not in refs:
                refs.append(c)
    stig_modl = model.ModelHeader.parse(stig.bundle.first("Modl"))
    n_mesh = sum(1 for b in blobs if b.tag == "Mesh")
    rest = bytes([1, 2, 0xFF, 0]) + struct.pack("<H", lod_union)
    modl = model.ModelHeader(n_mesh, 4, 2, len(md6.materials), rest)
    blobs.append(Blob("Modl", 1, 1, [Meta("BBox", 0, bbox.tobytes()), Meta("TRef", 0, struct.pack("<I%dI" % len(refs), len(refs), *refs))],
                      modl.serialize()))
    print("model: %d vertices, %d triangles, %d meshes (%d materials), stig modl was %s" % (nv, len(I) // 3, n_mesh, len(md6.materials), stig_modl.rest.hex()))
    return Bundle(1, 0, blobs).serialize(), (P, N, I, UV, J, W)


# ------------------------------------------------------------------------------------------------ package
def main():
    md6 = fh6_source.load()
    stig = geometry.ModelData(os.path.join(STIG, "driver.modelbin"))
    tex_params, tex_files = build_textures(md6, BUILD)
    modelbin, _ = build_model(md6, stig, tex_params)

    pkg = os.path.join(BUILD, "Master Chief")
    if os.path.isdir(pkg):
        shutil.rmtree(pkg)
    os.makedirs(os.path.join(pkg, "media", "drivers"))
    zpath = os.path.join(pkg, "media", "drivers", DRIVER + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("driver.modelbin", modelbin)
        for f in ("driver.gr2", "driver.gsf"):
            z.write(os.path.join(STIG, f), f)
        for name, data in sorted(tex_files.items()):
            z.writestr(name, data)
    with open(os.path.join(BUILD, "driver.modelbin"), "wb") as f:
        f.write(modelbin)
    manifest = {"format_version": 1, "id": "local-masterchief", "name": "Master Chief", "author": "Charelene",
                "version": "1.0.0", "description": "Master Chief as a Player Character, ported from Forza Horizon 6's outfit.",
                "database": "database.json", "media": "media", "dependencies": []}
    db = {"format_version": 2, "target": {"database": "media/stripped/gamedbRC.slt", "base_sha256": GAMEDB_SHA256},
          "policy": {"id_conflicts": "nearest", "update_conflicts": "last_wins"},
          "tables": {"DriverModels": {"key": ["Id"], "insert": [{"Id": 2, "MediaName": DRIVER}]}}}
    with open(os.path.join(pkg, "mod.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(pkg, "database.json"), "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print("package:", pkg, "zip", os.path.getsize(zpath), "bytes")


if __name__ == "__main__":
    main()
