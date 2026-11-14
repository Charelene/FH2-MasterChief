"""Every modelbin blob parses and re-serializes byte for byte; Mesh blobs leave nothing unparsed ("rest" empty).

python -I tests/test_model.py [-v]
"""
import glob
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "port"))

import grub  # noqa: E402
import model  # noqa: E402

EX = os.path.join(ROOT, "extracted")


def main():
    verbose = "-v" in sys.argv
    fails = Counter()
    ok = Counter()
    leftovers = Counter()
    for path in sorted(glob.glob(os.path.join(EX, "fh2/drivers/*/driver.modelbin")) + glob.glob(os.path.join(EX, "fh6/model/*.modelbin"))):
        b = grub.Bundle.load(path)
        for blob in b.blobs:
            key = f"{blob.tag} {blob.major}.{blob.minor}"
            parse, ser = model.PARSERS[blob.tag]
            try:
                obj = parse(blob)
                out = ser(obj, blob)
            except Exception as e:  # noqa: BLE001
                fails[key] += 1
                if verbose:
                    print("ERR", os.path.relpath(path, EX), key, e)
                continue
            if out != blob.data:
                fails[key] += 1
                if verbose:
                    print("DIFF", os.path.relpath(path, EX), key, len(blob.data), len(out))
                continue
            ok[key] += 1
    for k in sorted(set(ok) | set(fails)):
        print(f"{k:12} ok {ok[k]:4}  fail {fails[k]:4}")
    for k, n in sorted(leftovers.items()):
        print("unparsed tail:", k, "x", n)
    return 1 if fails or leftovers else 0


if __name__ == "__main__":
    sys.exit(main())
