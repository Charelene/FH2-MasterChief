"""The FH6 Master Chief driver model and its textures, as plain parts."""
import glob
import os
import re

import geometry
import swatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "extracted")
MODEL = os.path.join(EX, "fh6", "model", "Outfit_Armour_M_Driver.modelbin")


def load():
    return geometry.ModelData(MODEL)


def material_textures(md):
    """FH6 material id -> {slot: extracted swatchbin path} for slots bclr, nrml, rough_metal, emis."""
    out = {}
    for mid, blob in md.materials.items():
        found = {}
        for m in re.finditer(rb"forte_a(?:r)?mour_?([a-z_]*?)_(bclr|nrml|rough_metal|emis)_[0-9a-f-]{36}\.swatchbin", blob.data, re.I):
            base = m.group(0).decode().rsplit("\\", 1)[-1]
            hits = [p for p in glob.glob(os.path.join(EX, "fh6", "tex", "*.swatchbin")) if os.path.basename(p).lower() == base.lower()]
            if hits:
                found.setdefault(m.group(2).decode().lower(), hits[0])
        out[mid] = found
    return out


_cache = {}


def texture_rgba(path):
    if path not in _cache:
        _cache[path] = swatch.to_rgba(swatch.read_fh6(path))
    return _cache[path]
