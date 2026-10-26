"""Shared settings and helpers for the cross-lingual typographic study."""
import glob
import os
import unicodedata

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder
IMG_DIR = os.path.join(ROOT, "data", "imagenet-sample-images")
FONT_PATH = os.path.join(ROOT, "fonts", "DejaVuSans.ttf")       # DejaVu Sans 2.37; covers á é í ó ú ñ ç ã õ à è ò ï
FONT_PX = 28            # base font size on a 224 x 224 image
MIN_FONT_PX = 14
MAX_TEXT_W = 200        # long words are shrunk until they fit this width
IMG_PX = 224

# key: (open_clip architecture, checkpoint file, training corpus)
MODELS = {
    "vitb32_laion400m": ("ViT-B-32-quickgelu", "vit_b_32-quickgelu-laion400m_e32.pt", "LAION-400M"),
    "rn50_yfcc15m": ("RN50-quickgelu", "rn50-quickgelu-yfcc15m-455df137.pt", "YFCC15M"),
    "rn50_cc12m": ("RN50-quickgelu", "rn50-quickgelu-cc12m-f000538c.pt", "CC12M"),
    "rn101_yfcc15m": ("RN101-quickgelu", "rn101-quickgelu-yfcc15m-3e04b30e.pt", "YFCC15M"),
    "vitb16_laion400m": ("ViT-B-16", "vit_b_16-laion400m_e32-55e67d44.pt", "LAION-400M"),
    # OpenAI's own RN50 (TorchScript archive, trained on OpenAI's 400M web pairs). Kept in its own
    # folder; its SHA-256 must equal the hash in OpenAI's download URL before it is loaded.
    "rn50_openai": ("RN50", "openai_rn50/RN50.pt", "OpenAI WIT-400M"),
}
OPENAI_RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"


def image_files():
    """The 1,000 ImageNet sample photos, sorted by WordNet ID, which matches the class index order."""
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.JPEG")))
    assert len(files) == 1000, len(files)
    return files


def load_source(path):
    return Image.open(path).convert("RGB").resize((IMG_PX, IMG_PX))


_FONTS = {}


def font(px):
    if px not in _FONTS:
        _FONTS[px] = ImageFont.truetype(FONT_PATH, px)
    return _FONTS[px]


def fit_font(text):
    """Largest font size (<= FONT_PX) at which the text fits MAX_TEXT_W pixels."""
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    px = FONT_PX
    while px > MIN_FONT_PX:
        box = probe.textbbox((0, 0), text, font=font(px))
        if box[2] - box[0] <= MAX_TEXT_W:
            break
        px -= 1
    return px


def overlay(img, text):
    """Black word on a white box, centered horizontally, at 62% of the image height."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    px = fit_font(text)
    f = font(px)
    box = draw.textbbox((0, 0), text, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = (IMG_PX - tw) // 2, int(IMG_PX * 0.62)
    pad = 6
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad + 4], fill="white")
    draw.text((x - box[0], y - box[1]), text, fill="black", font=f)
    return img


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
            prev, d[j] = d[j], cur
    return d[len(b)]


def ortho_sim(word, english, accents=True):
    """1 - normalized Levenshtein distance; accents=False treats á as a, ñ as n, and so on."""
    a, b = word.lower(), english.lower()
    if not accents:
        a, b = strip_accents(a), strip_accents(b)
    return 1 - lev(a, b) / max(len(a), len(b))
