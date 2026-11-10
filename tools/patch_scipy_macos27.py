import argparse
import re
import subprocess
import sys
from pathlib import Path

SHIM = '''
class _UnavailableFortranExtension:
    def __init__(self, name):
        object.__setattr__(self, "_extension_name", name)

    def __getattr__(self, key):
        if key.startswith("__"):
            raise AttributeError(key)
        name = object.__getattribute__(self, "_extension_name")
        return _UnavailableFortranExtension(f"{name}.{key}")

    def __call__(self, *args, **kwargs):
        name = object.__getattribute__(self, "_extension_name")
        raise ImportError(f"{name} is unavailable: the macOS 27 dynamic loader rejects this "
                          "scipy Fortran extension")
'''

PATCHES = {
    "sparse/linalg/_svdp.py": (
        r"from \._propack import _spropack[^\n]*\n"
        r"from \._propack import _dpropack[^\n]*\n"
        r"from \._propack import _cpropack[^\n]*\n"
        r"from \._propack import _zpropack[^\n]*\n",
        SHIM + """
try:
    from ._propack import _spropack, _dpropack, _cpropack, _zpropack
except ImportError:
    _spropack = _UnavailableFortranExtension("_spropack")
    _dpropack = _UnavailableFortranExtension("_dpropack")
    _cpropack = _UnavailableFortranExtension("_cpropack")
    _zpropack = _UnavailableFortranExtension("_zpropack")
""",
    ),
    "integrate/_ode.py": (
        r"from \. import _vode\nfrom \. import _dop\nfrom \. import _lsoda\n",
        SHIM + """
try:
    from . import _vode, _dop, _lsoda
except ImportError:
    _vode = _UnavailableFortranExtension("_vode")
    _dop = _UnavailableFortranExtension("_dop")
    _lsoda = _UnavailableFortranExtension("_lsoda")
""",
    ),
    "optimize/_cobyla_py.py": (
        r"from scipy\.optimize import _cobyla as cobyla\n",
        SHIM + """
try:
    from scipy.optimize import _cobyla as cobyla
except ImportError:
    cobyla = _UnavailableFortranExtension("_cobyla")
""",
    ),
}

MODULES = ["scipy.ndimage", "scipy.spatial", "scipy.sparse.linalg", "scipy.stats",
           "skimage.transform", "sklearn.metrics", "pylidc"]

VERIFY = ("import warnings\nwarnings.filterwarnings('ignore')\n"
          f"mods = {MODULES!r}\n"
          "bad = []\n"
          "for m in mods:\n"
          "    try:\n"
          "        __import__(m)\n"
          "        print('  OK   ' + m)\n"
          "    except Exception as e:\n"
          "        bad.append(m)\n"
          "        print('  FAIL ' + m + ': ' + str(e)[:80])\n"
          "raise SystemExit(1 if bad else 0)\n")


def import_chain_ok(show: bool = True) -> bool:
    r = subprocess.run([sys.executable, "-c", VERIFY], capture_output=True, text=True)
    if show:
        print(r.stdout.rstrip())
        if r.returncode and r.stderr:
            print(r.stderr[-400:])
    return r.returncode == 0


def scipy_root() -> Path:
    import importlib.util
    spec = importlib.util.find_spec("scipy")
    if spec is None or spec.origin is None:
        raise SystemExit("scipy is not installed in this environment")
    return Path(spec.origin).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    print("checking the import chain:")
    if import_chain_ok():
        print("\nscipy and its dependants import cleanly; nothing to patch")
        return 0

    root = scipy_root()
    print(f"\nscipy at {root}")
    for rel, (pattern, replacement) in PATCHES.items():
        path = root / rel
        if not path.exists():
            print(f"  missing            {rel}")
            continue
        text = path.read_text()
        match = re.search(pattern, text)
        if match is None:
            print(f"  pattern not found  {rel}")
            continue
        if a.check:
            print(f"  would patch        {rel}")
            continue
        path.with_suffix(path.suffix + ".orig").write_text(text)
        path.write_text(text[:match.start()] + replacement + text[match.end():])
        print(f"  patched            {rel}  (original kept as {path.name}.orig)")

    if a.check:
        return 0
    print("\nverifying the import chain:")
    return 0 if import_chain_ok() else 1


if __name__ == "__main__":
    sys.exit(main())
