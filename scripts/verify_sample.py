"""Recompute a random sample of stimuli for one model and compare with the shipped results.

This checks a local setup (photographs, font, checkpoints, and package versions) in minutes,
without rerunning the full experiment.
Usage: python3 scripts/verify_sample.py rn50_yfcc15m [n_stimuli]
"""
import csv
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import numpy as np  # noqa: E402

from common import MODELS, image_files, load_source, overlay  # noqa: E402
from run_models import log_probs, load_model  # noqa: E402


def main(key, n):
    if key not in MODELS:
        sys.exit(f"unknown model {key}; choose from {sorted(MODELS)}")
    with open(os.path.join(ROOT, "stimuli", "manifest.csv"), newline="", encoding="utf-8") as fh:
        manifest = {int(m["stim_id"]): m for m in csv.DictReader(fh)}
    with open(os.path.join(ROOT, "results", f"{key}_stimuli.csv"), newline="") as fh:
        shipped = {int(r["stim_id"]): r for r in csv.DictReader(fh)}
    sample = random.Random(0).sample(sorted(shipped), n)
    files = image_files()
    model, pre, text, scale = load_model(key)
    clean = np.load(os.path.join(ROOT, "results", f"{key}_clean.npz"))["logp"]
    worst, agree = 0.0, 0
    for start in range(0, n, 32):
        ids = sample[start:start + 32]
        rows = [manifest[i] for i in ids]
        lp = log_probs(model, pre, text, scale,
                       [overlay(load_source(files[int(m["source_idx"])]), m["word"]) for m in rows]).numpy()
        for i, m, row in zip(ids, rows, lp):
            t, s = int(m["target_idx"]), int(m["source_idx"])
            ref = shipped[i]
            worst = max(worst, abs(float(row[t]) - float(ref["target_logp"])),
                        abs(float(row[t] - clean[s, t]) - float(ref["d_target_logp"])))
            agree += int(int(row.argmax()) == int(ref["pred"]))
    print(f"{key}: {n} stimuli, predictions identical for {agree}, largest log-probability difference {worst:.5f}")
    if agree < n or worst > 1e-3:
        sys.exit("results differ from the shipped files beyond rounding")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rn50_yfcc15m", int(sys.argv[2]) if len(sys.argv) > 2 else 64)
