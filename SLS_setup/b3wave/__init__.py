"""
b3wave: the B3-wave training / evaluation package.

Everything here is additive: nothing in main_train.py, main_eval.py, utils/ or model/ is changed,
so B0 / B1 / B2 runs stay reproducible from the original scripts. `python -u -m b3wave.train` with a
B2e command line reproduces B2e (every new flag defaults to the old behaviour); see README.md.
"""

__version__ = "0.1.0"
