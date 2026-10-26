"""
b4wave: the B4 research cycle package (rdloop/B4).

Additive: nothing in SLS_setup/b3wave/, model/, utils/ or main_*.py is changed. `python -u -m b4wave.train` with the base command line reproduces the base (every new flag defaults to the old behaviour); see README.md.
"""

import hashlib
from pathlib import Path

__version__ = "0.2.0"

MODULE_DIR = Path(__file__).resolve().parent


def module_files():
    """The module's own .py files (top level, no __pycache__), sorted by file name."""
    return sorted((p for p in MODULE_DIR.glob("*.py") if p.is_file()), key=lambda p: p.name)


def module_sha256():
    """
    sha256 over the module's .py files sorted by name: for each file, its name, a NUL byte, the
    sha256 hex digest of its bytes and a newline. Written to config.yaml as `module_sha256` and
    printed as `b4wave sha256: <first 12>` so every run names the exact code it ran.
    """
    h = hashlib.sha256()
    for p in module_files():
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
        h.update(b"\n")
    return h.hexdigest()
