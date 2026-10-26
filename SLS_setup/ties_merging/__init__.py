"""
ties_merging: TIES-Merging (Yadav et al. 2023, arXiv 2306.01708) for plain ModelSLS checkpoints.

Additive: nothing in SLS_setup/main_*.py, model/, utils/, b3wave/, b4wave/ or the existing
scripts is changed. The merged checkpoint is a plain ModelSLS state_dict, so local_eval.py
--model_path, make_submission.py and /post-train-eval work on it unchanged. See README.md.
"""

import hashlib
from pathlib import Path

__version__ = "0.1.0"

MODULE_DIR = Path(__file__).resolve().parent


def module_files():
    """The module's own .py files (top level, no __pycache__), sorted by file name."""
    return sorted((p for p in MODULE_DIR.glob("*.py") if p.is_file()), key=lambda p: p.name)


def module_sha256():
    """sha256 over the module's .py files (name, NUL, file sha256, newline), as in b4wave."""
    h = hashlib.sha256()
    for p in module_files():
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
        h.update(b"\n")
    return h.hexdigest()
