"""Run the stimulus set through the CLIP models, resumably, within a time budget.

Usage: python3 code/run_models.py [max_minutes]
Models run in QUEUE order. Each writes results/<model>_clean.npz (clean log-probabilities for
all 1,000 photos) and results/<model>_stimuli.csv (one row per stimulus). Re-running resumes
where the previous run stopped; the script prints DONE when every model has finished.
"""
import csv
import hashlib
import os
import sys
import time

# Explicit code path so the script also works under `python3 -I` (isolated mode).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import open_clip
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES

from common import MODELS, OPENAI_RN50_SHA256, ROOT, XLMR_SHA256, create_openclip, image_files, load_source, overlay

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))   # results do not depend on this
QUEUE = [("vitb32_laion400m", 16), ("rn50_yfcc15m", 10), ("rn50_cc12m", 10),
         ("vitb16_laion400m", 10), ("rn101_yfcc15m", 10), ("rn50_openai", 10),
         ("vitb32_xlmr_laion5b", 16)]      # (model, photos per target); the multilingual ViT-B/32 matches its twin
BATCH = 64
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
FIELDS = ["stim_id", "pred", "target_logp", "target_rank", "true_logp", "d_target_logp", "d_true_logp"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(key):
    arch, ckpt, _ = MODELS[key]
    path = os.path.join(ROOT, "weights", ckpt)
    if key == "rn50_openai":
        # OpenAI checkpoints are TorchScript archives; refuse to load anything but the published file.
        digest = sha256(path)
        if digest != OPENAI_RN50_SHA256:
            raise SystemExit(f"[{key}] checksum mismatch ({digest}); not loading")
        model = open_clip.openai.load_openai_model(path, precision="fp32", device="cpu")
        pre = open_clip.image_transform(model.visual.image_size, is_train=False)
        tok = open_clip.get_tokenizer(arch)
    else:
        if key == "vitb32_xlmr_laion5b" and sha256(path) != XLMR_SHA256:
            raise SystemExit(f"[{key}] checksum mismatch; not loading")
        model, pre, tok = create_openclip(key)
    model.eval()
    with torch.no_grad():
        t = model.encode_text(tok([f"a photo of a {n}." for n in IMAGENET_CLASSNAMES]))
        t = t / t.norm(dim=-1, keepdim=True)
    return model, pre, t, model.logit_scale.exp().item()


def log_probs(model, pre, text, scale, pils):
    x = torch.stack([pre(p) for p in pils])
    with torch.no_grad():
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return torch.log_softmax(scale * f @ text.T, dim=-1)


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {int(r["stim_id"]) for r in csv.DictReader(fh)}


def run_model(key, max_rank, deadline, sources, manifest):
    out_path = os.path.join(RES, f"{key}_stimuli.csv")
    todo = [m for m in manifest if int(m["source_rank"]) < max_rank]
    finished = done_ids(out_path)
    todo = [m for m in todo if int(m["stim_id"]) not in finished]
    if not todo:
        return True
    if not os.path.exists(os.path.join(ROOT, "weights", MODELS[key][1])):
        print(f"[{key}] WAITING: checkpoint not in weights/ yet", flush=True)
        return False
    model, pre, text, scale = load_model(key)

    clean_path = os.path.join(RES, f"{key}_clean.npz")
    if os.path.exists(clean_path):
        clean = np.load(clean_path)["logp"]
    else:
        parts = [log_probs(model, pre, text, scale, sources[i:i + BATCH]).numpy()
                 for i in range(0, 1000, BATCH)]
        clean = np.concatenate(parts)
        np.savez_compressed(clean_path, logp=clean)
        acc = float((clean.argmax(1) == np.arange(1000)).mean())
        print(f"[{key}] clean zero-shot top-1 on the 1,000 photos: {acc:.3f}", flush=True)

    new_file = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
    cache = {}            # (target, word, source) -> result row; duplicates share images
    t0, n_new = time.time(), 0
    for i in range(0, len(todo), BATCH):
        if time.time() > deadline:
            fh.close()
            print(f"[{key}] paused after {n_new} new stimuli ({n_new / max(time.time() - t0, 1):.1f}/s)", flush=True)
            return False
        batch = todo[i:i + BATCH]
        keys = [(m["target_idx"], m["word"], m["source_idx"]) for m in batch]
        need = [k for k in dict.fromkeys(keys) if k not in cache]
        if need:
            lp = log_probs(model, pre, text, scale, [overlay(sources[int(s)], w) for _, w, s in need]).numpy()
            for k, row in zip(need, lp):
                t, s = int(k[0]), int(k[2])
                cache[k] = {
                    "pred": int(row.argmax()),
                    "target_logp": round(float(row[t]), 4),
                    "target_rank": int((row > row[t]).sum()),
                    "true_logp": round(float(row[s]), 4),
                    "d_target_logp": round(float(row[t] - clean[s, t]), 4),
                    "d_true_logp": round(float(row[s] - clean[s, s]), 4),
                }
            n_new += len(need)
        for m, k in zip(batch, keys):
            writer.writerow({"stim_id": m["stim_id"], **cache[k]})
        fh.flush()
        if len(cache) > 5000:
            cache.clear()
        if (i // BATCH) % 50 == 0:
            print(f"[{key}] {i + len(batch)}/{len(todo)} rows this session, "
                  f"{n_new / max(time.time() - t0, 1):.1f} images/s", flush=True)
    fh.close()
    print(f"[{key}] finished", flush=True)
    return True


def main():
    max_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
    deadline = time.time() + 60 * max_minutes
    files = image_files()
    sources = [load_source(f) for f in files]
    with open(os.path.join(ROOT, "stimuli", "manifest.csv"), newline="", encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))
    for key, max_rank in QUEUE:
        if not run_model(key, max_rank, deadline, sources, manifest):
            print("PARTIAL", flush=True)
            return
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
