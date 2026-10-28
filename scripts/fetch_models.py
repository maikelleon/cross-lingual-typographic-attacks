"""Download the nine CLIP checkpoints from their official release locations and verify them.

The OpenCLIP checkpoints come from the OpenCLIP release assets, the ResNet-50, ViT-B/32, and ViT-L/14
released by OpenAI from OpenAI's model server, and the multilingual ViT-B/32 (LAION-5B, XLM-RoBERTa
text encoder) from LAION's Hugging Face repository, together with the configuration and tokenizer of
its text encoder, at pinned revisions. Files are saved under weights/ with the names that
code/common.py expects and are checked against checksums/model_checkpoints.sha256.
Usage: python3 scripts/fetch_models.py [--verify-only]
"""
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENCLIP = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
XLMR_HUB = ("https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k/resolve/"
            "506d40eb551f4801a1c27fc20a31c7b8f590deda/")
SOURCES = {
    "vit_b_32-quickgelu-laion400m_e32.pt": OPENCLIP + "vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
    "vit_b_16-laion400m_e32-55e67d44.pt": OPENCLIP + "vit_b_16-laion400m_e32-55e67d44.pt",
    "rn50-quickgelu-yfcc15m-455df137.pt": OPENCLIP + "rn50-quickgelu-yfcc15m-455df137.pt",
    "rn101-quickgelu-yfcc15m-3e04b30e.pt": OPENCLIP + "rn101-quickgelu-yfcc15m-3e04b30e.pt",
    "rn50-quickgelu-cc12m-f000538c.pt": OPENCLIP + "rn50-quickgelu-cc12m-f000538c.pt",
    "openai_rn50/RN50.pt": "https://openaipublic.azureedge.net/clip/models/"
                           "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "openai_vitl14/ViT-L-14.pt": "https://openaipublic.azureedge.net/clip/models/"
                                 "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "openai_vitb32/ViT-B-32.pt": "https://openaipublic.azureedge.net/clip/models/"
                                 "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "xlmr_vitb32_laion5b/open_clip_pytorch_model.bin": XLMR_HUB + "open_clip_pytorch_model.bin",
    "xlm_roberta_base/tokenizer.json": XLMR_HUB + "tokenizer.json",
    "xlm_roberta_base/config.json": "https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/"
                                    "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/config.json",
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
