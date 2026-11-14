"""Install or remove the built "Master Chief" Extra Options package in an Azimuth folder (EO 0.6.0 layout).

python -I port/install.py --azimuth <Azimuth folder>               install / update
python -I port/install.py --azimuth <Azimuth folder> --uninstall

Copies build/Master Chief to <azimuth>/mods/Forza Horizon 2/Master Chief and lists it last in that folder's
loadorder.txt (the previous file is kept as loadorder.txt.bak-masterchief). Refuses while azimuth.exe runs.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_NAME = "Master Chief"
PKG = os.path.join(ROOT, "build", PKG_NAME)


def game_running(azimuth_dir):
    """True when an azimuth.exe from this folder runs (other Azimuth folders, e.g. test rigs, do not count)."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Process azimuth -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }"],
                       capture_output=True, text=True)
    want = os.path.normcase(os.path.abspath(os.path.join(azimuth_dir, "azimuth.exe")))
    return any(os.path.normcase(os.path.abspath(p.strip())) == want for p in r.stdout.splitlines() if p.strip())


def read_order(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return [ln.rstrip("\r\n") for ln in f if ln.strip()]


def write_order(path, lines):
    eol = "\r\n"  # Extra Options writes CRLF; keep a file's own endings when it has LF only
    if os.path.exists(path):
        with open(path, "rb") as f:
            raw = f.read()
        if b"\n" in raw and b"\r\n" not in raw:
            eol = "\n"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("".join(ln + eol for ln in lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--azimuth", required=True)
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args()
    mods = os.path.join(a.azimuth, "mods", "Forza Horizon 2")
    if not os.path.isfile(os.path.join(a.azimuth, "winmm.dll")) or not os.path.isdir(mods):
        sys.exit(f"{a.azimuth}: no Extra Options 0.6.0 layout (winmm.dll, mods\\Forza Horizon 2)")
    if game_running(a.azimuth):
        sys.exit("azimuth.exe is running; close the game first")
    dest = os.path.join(mods, PKG_NAME)
    order_path = os.path.join(mods, "loadorder.txt")
    order = read_order(order_path)
    if os.path.exists(order_path) and not os.path.exists(order_path + ".bak-masterchief"):
        shutil.copy2(order_path, order_path + ".bak-masterchief")
    if a.uninstall:
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        write_order(order_path, [ln for ln in order if ln.strip() != PKG_NAME])
        print("removed", dest)
        return
    if not os.path.isfile(os.path.join(PKG, "mod.json")):
        sys.exit("build the package first: python -I port/fh2_build.py")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(PKG, dest)
    if PKG_NAME not in [ln.strip() for ln in order]:
        order.append(PKG_NAME)
        write_order(order_path, order)
    print("installed", dest)
    print("load order:", ", ".join(order))


if __name__ == "__main__":
    main()
