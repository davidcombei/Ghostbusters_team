"""Fake b3wave.checks for the rdloop tests."""
import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument("what", choices=["augmenter", "data", "sampler", "losses", "model", "optim", "all"])
p.add_argument("--ckpt", default=None)
a = p.parse_args()
print(f"PASS {a.what}")
sys.exit(0)
