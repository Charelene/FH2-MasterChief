"""Round trips: every FH2/FH6 bundle we touch must re-serialize byte for byte.

python -I tests/test_grub.py
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "port"))

import grub  # noqa: E402

EX = os.path.join(ROOT, "extracted")


def files():
    pats = ["fh2/drivers/*/driver.modelbin", "fh2/cartex/driver/swatches/*.swatchbin", "fh2/carmat/driver/*.materialbin",
            "fh6/model/*.modelbin", "fh6/tex/*.swatchbin"]
    for p in pats:
        yield from sorted(glob.glob(os.path.join(EX, p)))


def main():
    bad = 0
    n = 0
    for path in files():
        raw = open(path, "rb").read()
        try:
            out = grub.Bundle.parse(raw).serialize()
        except Exception as e:  # noqa: BLE001
            print("FAIL", os.path.relpath(path, EX), "error", e)
            bad += 1
            continue
        n += 1
        if out != raw:
            first = next((i for i in range(min(len(out), len(raw))) if out[i] != raw[i]), min(len(out), len(raw)))
            print("FAIL", os.path.relpath(path, EX), f"len {len(raw)} -> {len(out)}, first diff at 0x{first:x}")
            bad += 1
    print(f"{n - bad}/{n} bundles round-trip byte-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
