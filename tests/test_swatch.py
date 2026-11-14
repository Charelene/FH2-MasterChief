"""FH2 swatchbins: detile with xg.dll, retile, rewrite -> byte-identical to the game's file. Also decodes a few to PNG
for a look (build/preview/tex_*.png).

python -I tests/test_swatch.py
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "port"))

import png  # noqa: E402
import swatch  # noqa: E402

EX = os.path.join(ROOT, "extracted")
OUT = os.path.join(ROOT, "build", "preview")


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(EX, "fh2/cartex/driver/swatches/*.swatchbin")))
    bad = 0
    for path in files:
        raw = open(path, "rb").read()
        hdr, tex = swatch.read_fh2(path)
        guid = __import__("uuid").UUID(bytes_le=hdr.guid)
        out = swatch.write_fh2(None, tex, hdr, guid)
        if out != raw:
            bad += 1
            print("DIFF", os.path.basename(path), hdr.width, hdr.height, hdr.fmt, f"flags {hdr.flags:#010x}")
    print(f"{len(files) - bad}/{len(files)} FH2 swatchbins re-tile byte-identical")
    for name in ("stigsuitmat_diff", "stighelmet_diff", "stigsuitmat_glos", "stighelmet_nrml"):
        p = glob.glob(os.path.join(EX, f"fh2/cartex/driver/swatches/{name}_*.swatchbin"))[0]
        hdr, tex = swatch.read_fh2(p)
        png.write(os.path.join(OUT, f"fh2_{name}.png"), swatch.to_rgba(tex))
        print(name, tex.width, tex.height, "fmt", tex.fmt, f"flags {hdr.flags:#010x}")
    for name in ("Forte_Armour_BCLR", "Forte_Armour_Visor_BCLR", "Forte_Armour_NRML", "Forte_Armour_ROUGH_METAL", "Forte_Armour_EMIS"):
        p = glob.glob(os.path.join(EX, f"fh6/tex/{name}_*.swatchbin"))[0]
        tex = swatch.read_fh6(p)
        png.write(os.path.join(OUT, f"fh6_{name}.png"), swatch.to_rgba(tex))
        print(name, tex.width, tex.height, "fmt", tex.fmt, "mips", len(tex.mips))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
