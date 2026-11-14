"""Preview sheets: build/preview/source.png (FH6 as shipped), retargeted.png (on FH2's skeleton, FH2 joints in red),
stig.png (FH2's Stig for scale)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I leaves the script dir out

import numpy as np  # noqa: E402

import fh6_source
import geometry
import preview
import retarget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "preview")


def items_for(parts, md, positions=None, normals=None):
    tex = fh6_source.material_textures(md) if md is not None else {}
    out = []
    for k, p in enumerate(parts):
        t = tex.get(p.material, {}).get("bclr")
        out.append({"positions": p.positions if positions is None else positions[k],
                    "normals": p.normals if normals is None else normals[k], "tris": p.tris,
                    "uv": p.uvs.get(0), "texture": fh6_source.texture_rgba(t) if t else None, "color": (150, 150, 150)})
    return out


def main(lod_mask=2):
    os.makedirs(OUT, exist_ok=True)
    md6 = fh6_source.load()
    parts = [p for p in md6.parts() if p.lod_mask & lod_mask]
    preview.sheet(os.path.join(OUT, "source.png"), items_for(parts, md6))
    stig = geometry.ModelData(os.path.join(fh6_source.EX, "fh2", "drivers", "stig", "driver.modelbin"))
    rt = retarget.Retarget(md6.skeleton, stig.skeleton)
    pos, nrm = [], []
    for p in parts:
        a, b, _ = rt.deform(p.positions, p.normals, p.tangents, p.joints, p.weights)
        pos.append(a)
        nrm.append(b)
    joints2 = geometry.world_matrices(stig.skeleton)[:, 3, :3]
    allp = np.concatenate(pos)
    bounds = (np.minimum(allp.min(0), -0.9) , np.maximum(allp.max(0), 0.9))
    preview.sheet(os.path.join(OUT, "retargeted.png"), items_for(parts, md6, pos, nrm), joints=joints2, bounds=bounds)
    sp = [p for p in stig.parts() if p.lod_mask & 2]
    preview.sheet(os.path.join(OUT, "stig.png"), [{"positions": p.positions, "normals": p.normals, "tris": p.tris, "uv": None,
                                                    "texture": None, "color": (200, 200, 200)} for p in sp], joints=joints2, bounds=bounds)
    print("wrote", OUT)


if __name__ == "__main__":
    main(int(sys.argv[1], 0) if len(sys.argv) > 1 else 2)
