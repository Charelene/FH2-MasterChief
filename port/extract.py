"""Copy the source files out of the two games into extracted/ (read-only on the game folders).

python -I port/extract.py [--fh2 S:/Forza/Games/ForzaHorizon2/Mount] [--fh6 S:/Forza/Games/Forza.Horizon.6]
"""
import argparse
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "extracted")


def unzip(src, dst, keep=lambda name: True, flat=False):
    n = 0
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if name.endswith("/") or not keep(name):
                continue
            path = os.path.join(dst, os.path.basename(name) if flat else name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(z.read(info))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fh2", default="S:/Forza/Games/ForzaHorizon2/Mount")
    ap.add_argument("--fh6", default="S:/Forza/Games/Forza.Horizon.6")
    a = ap.parse_args()

    outfit = os.path.join(a.fh6, "media/Cinematic_Assets/Characters/Garments/Outfit")
    for name in ("Outfit_Armour_M_Driver", "Outfit_Armour_M"):
        print(name, unzip(os.path.join(outfit, name + ".zip"), os.path.join(OUT, "fh6/model")))
    print("fh6 textures", unzip(os.path.join(a.fh6, "media/_library/TexturesPG/characters.zip"), os.path.join(OUT, "fh6/tex"),
                                lambda n: os.path.basename(n).lower().startswith(("forte_armour", "forte_amour")), flat=True))

    for name in ("stig", "default", "defaultRace", "AIRace", "AIUnsuited", "suited", "unsuited"):
        print(name, unzip(os.path.join(a.fh2, "media/drivers", name + ".zip"), os.path.join(OUT, "fh2/drivers", name)))
    lib = os.path.join(a.fh2, "media/cars/_library")
    print("fh2 driver textures", unzip(os.path.join(lib, "Textures.zip"), os.path.join(OUT, "fh2/cartex"), lambda n: n.lower().startswith("driver/")))
    print("fh2 driver materials", unzip(os.path.join(lib, "Materials.zip"), os.path.join(OUT, "fh2/carmat"), lambda n: n.lower().startswith("driver/")))


if __name__ == "__main__":
    main()
