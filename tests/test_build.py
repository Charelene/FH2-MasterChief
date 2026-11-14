"""Read the built MasterChief.zip back the way FH2 would: parse driver.modelbin with the FH2 decoder, check every
reference (materials, textures in the zip, buffers, TRef), decode the tiled textures and render
build/preview/built.png (front | side | back) and built_lods.png (LODS cockpit set vs LOD0).

python -I tests/test_build.py
"""
import os
import re
import struct
import sys
import zipfile
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "port"))

import numpy as np  # noqa: E402

import geometry  # noqa: E402
import grub  # noqa: E402
import model  # noqa: E402
import png  # noqa: E402
import preview  # noqa: E402
import swatch  # noqa: E402

PKG = os.path.join(ROOT, "build", "Master Chief")
OUT = os.path.join(ROOT, "build", "preview")


def main():
    problems = []
    z = zipfile.ZipFile(os.path.join(PKG, "media", "drivers", "MasterChief.zip"))
    names = {n.lower(): n for n in z.namelist()}
    raw = z.read("driver.modelbin")
    if grub.Bundle.parse(raw).serialize() != raw:
        problems.append("driver.modelbin does not round-trip")
    md = geometry.ModelData(raw)
    stig = geometry.ModelData(os.path.join(ROOT, "extracted", "fh2", "drivers", "stig", "driver.modelbin"))
    if md.bundle.first("Skel").data != stig.bundle.first("Skel").data:
        problems.append("skeleton differs from the Stig's")
    # every blob version matches what FH2's own driver files use
    seen = {(b.tag, b.major, b.minor) for b in stig.bundle.blobs}
    for b in md.bundle.blobs:
        if (b.tag, b.major, b.minor) not in seen:
            problems.append(f"blob {b.tag} {b.major}.{b.minor} not used by FH2's Stig")
    # textures referenced by the materials exist in the zip and decode
    tex_imgs = {}
    paths = set()
    for mid, blob in md.materials.items():
        mt = model.material_bundle(blob).first("MTPR").data
        for m in re.finditer(rb"Game:[\x20-\x7e]+?\.swatchbin", mt):
            paths.add(m.group(0).decode())
    prefix = "game:\\media\\drivers\\masterchief\\"
    for p in sorted(paths):
        rel = p.lower()[len(prefix):].replace("\\", "/") if p.lower().startswith(prefix) else None
        if rel is None or rel not in names:
            problems.append("texture not in the zip: " + p)
            continue
        tmp = os.path.join(OUT, "_tmp.swatchbin")
        with open(tmp, "wb") as f:
            f.write(z.read(names[rel]))
        hdr, tex = swatch.read_fh2(tmp)
        tex_imgs[p.lower()] = swatch.to_rgba(tex)
    os.remove(os.path.join(OUT, "_tmp.swatchbin"))
    tref = md.bundle.first("Modl").get_meta("TRef").data
    refs = set(struct.unpack_from("<%dI" % struct.unpack_from("<I", tref)[0], tref, 4))
    if refs != {zlib.crc32(p.lower().encode()) for p in paths}:
        problems.append("TRef does not list exactly the material textures")
    # geometry: every part decodes, weights sum to 1, bones exist
    parts = md.parts()
    nb = len(md.skeleton.bones)
    for p in parts:
        if not np.allclose(p.weights.sum(1), 1, atol=2e-3):
            problems.append(f"{p.name}: weights do not sum to 1")
        if p.joints.max() >= nb:
            problems.append(f"{p.name}: bone index out of range")
        if not np.all(np.isfinite(p.positions)) or not np.all(np.isfinite(p.normals)):
            problems.append(f"{p.name}: non-finite vertex data")
    diff_of = {}
    for mid, blob in md.materials.items():
        mt = model.material_bundle(blob).first("MTPR").data
        d = [m.group(0).decode().lower() for m in re.finditer(rb"Game:[\x20-\x7e]+?_diff_[\x20-\x7e]+?\.swatchbin", mt)]
        diff_of[mid] = tex_imgs.get(d[0]) if d else None

    def items(sel):
        return [{"positions": p.positions, "normals": p.normals, "tris": p.tris, "uv": p.uvs.get(0),
                 "texture": diff_of.get(p.material), "color": (150, 150, 150)} for p in sel]
    lod0 = [p for p in parts if p.lod_mask == 2]
    lods = [p for p in parts if p.lod_mask == 1]
    J = geometry.world_matrices(md.skeleton)[:, 3, :3]
    preview.sheet(os.path.join(OUT, "built.png"), items(lod0), joints=J)
    b = (np.array([-0.9, 0.0, -0.3]), np.array([0.9, 1.75, 0.3]))
    png.write(os.path.join(OUT, "built_lods.png"), np.concatenate([preview.render(items(lods), "front", (600, 900), bounds=b),
                                                                  preview.render(items(lod0), "front", (600, 900), bounds=b)], axis=1))
    print(f"{len(parts)} visible meshes, {len(md.meshes)} meshes in all, {len(paths)} textures, {len(md.materials)} materials")
    for x in problems:
        print("PROBLEM", x)
    print("ok" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
