"""Download the six CLIP checkpoints from their official release locations and verify them.

The OpenCLIP checkpoints come from the OpenCLIP release assets and the ResNet-50 released by
OpenAI from OpenAI's model server. Files are saved under weights/ with the names that
code/common.py expects and are checked against checksums/model_checkpoints.sha256.
Usage: python3 scripts/fetch_models.py [--verify-only]
"""
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENCLIP = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
SOURCES = {
    "vit_b_32-quickgelu-laion400m_e32.pt": OPENCLIP + "vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
    "vit_b_16-laion400m_e32-55e67d44.pt": OPENCLIP + "vit_b_16-laion400m_e32-55e67d44.pt",
    "rn50-quickgelu-yfcc15m-455df137.pt": OPENCLIP + "rn50-quickgelu-yfcc15m-455df137.pt",
    "rn101-quickgelu-yfcc15m-3e04b30e.pt": OPENCLIP + "rn101-quickgelu-yfcc15m-3e04b30e.pt",
    "rn50-quickgelu-cc12m-f000538c.pt": OPENCLIP + "rn50-quickgelu-cc12m-f000538c.pt",
    "openai_rn50/RN50.pt": "https://openaipublic.azureedge.net/clip/models/"
                           "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(verify_only=False):
    expected = {}
    with open(os.path.join(ROOT, "checksums", "model_checkpoints.sha256"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                digest, name = line.split(None, 1)
                expected[name.strip()] = digest
    ok = True
    for name, url in SOURCES.items():
        path = os.path.join(ROOT, "weights", name)
        if not os.path.exists(path):
            if verify_only:
                print(f"missing  {name}")
                ok = False
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading {name}", flush=True)
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        good = sha256(path) == expected[name]
        ok &= good
        print(f"{'ok      ' if good else 'MISMATCH'} {name}")
    if not ok:
        sys.exit("some checkpoints are missing or differ from the ones used in the study")


if __name__ == "__main__":
    main(verify_only="--verify-only" in sys.argv)
