"""Download the third-party inputs that this repository does not redistribute, and verify them.

1. The 1,000 ImageNet sample photographs (one per class) from the public repository
   EliSchwartz/imagenet-sample-images at commit bc69043, into data/imagenet-sample-images/.
2. The NLTK WordNet and Open Multilingual Wordnet 2.0 corpora, into data/nltk_data/corpora/.

Every file is checked against the SHA-256 lists in checksums/.
Usage: python3 scripts/fetch_data.py
"""
import hashlib
import io
import os
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO_COMMIT = "bc69043e2862758a1906c6a24a792bb28c6bb2ff"
PHOTOS_URL = f"https://github.com/EliSchwartz/imagenet-sample-images/archive/{PHOTO_COMMIT}.zip"
PHOTO_RAW = f"https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/{PHOTO_COMMIT}/"
NLTK_URLS = {
    "wordnet.zip": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip",
    "omw-2.0.zip": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-2.0.zip",
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_checksums(name):
    sums = {}
    with open(os.path.join(ROOT, "checksums", name), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                digest, path = line.strip().split(None, 1)
                sums[path] = digest
    return sums


def download(url):
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def fetch_photos():
    out = os.path.join(ROOT, "data", "imagenet-sample-images")
    expected = read_checksums("imagenet_sample_photos.sha256")
    if os.path.isdir(out) and all(os.path.exists(os.path.join(out, f)) for f in expected):
        print("photos already present")
    else:
        os.makedirs(out, exist_ok=True)
        try:
            archive = zipfile.ZipFile(io.BytesIO(download(PHOTOS_URL)))
            for member in archive.namelist():
                name = os.path.basename(member)
                if name in expected:
                    with open(os.path.join(out, name), "wb") as fh:
                        fh.write(archive.read(member))
        except OSError as err:
            # Some networks block repository archives; fall back to fetching the files one by one.
            print(f"archive download failed ({err}); fetching photographs individually", flush=True)
            for k, name in enumerate(sorted(expected), start=1):
                path = os.path.join(out, name)
                if not os.path.exists(path):
                    with urllib.request.urlopen(PHOTO_RAW + urllib.parse.quote(name)) as resp, open(path, "wb") as fh:
                        fh.write(resp.read())
                if k % 100 == 0:
                    print(f"  {k}/{len(expected)}", flush=True)
    bad = [f for f, d in expected.items()
           if not os.path.exists(os.path.join(out, f)) or sha256_bytes(open(os.path.join(out, f), "rb").read()) != d]
    if bad:
        sys.exit(f"{len(bad)} photographs are missing or differ from the published set, e.g. {bad[:3]}")
    print(f"photos verified: {len(expected)} files")


def fetch_nltk():
    out = os.path.join(ROOT, "data", "nltk_data", "corpora")
    os.makedirs(out, exist_ok=True)
    expected = read_checksums("nltk_corpora.sha256")
    for name, url in NLTK_URLS.items():
        path = os.path.join(out, name)
        data = open(path, "rb").read() if os.path.exists(path) else download(url)
        if sha256_bytes(data) != expected[name]:
            sys.exit(f"{name} differs from the version used in the study")
        with open(path, "wb") as fh:
            fh.write(data)
        zipfile.ZipFile(io.BytesIO(data)).extractall(out)
        print(f"{name} verified and extracted")


if __name__ == "__main__":
    fetch_photos()
    fetch_nltk()
