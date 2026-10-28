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
    # OpenCLIP's multilingual model: the ViT-B/32 image encoder trained on LAION-5B with an XLM-RoBERTa text
    # encoder, which reads captions in many languages. Its text encoder is built from the xlm-roberta-base
    # configuration and tokenizer kept in weights/xlm_roberta_base, so loading needs no network access.
    "vitb32_xlmr_laion5b": ("xlm-roberta-base-ViT-B-32", "xlmr_vitb32_laion5b/open_clip_pytorch_model.bin", "LAION-5B"),
    # OpenAI's ViT-L/14 at 224 pixels (TorchScript archive, same corpus as the OpenAI RN50), the encoder
    # family used in many multimodal language models. Its SHA-256 is also checked before loading.
    "vitl14_openai": ("ViT-L-14", "openai_vitl14/ViT-L-14.pt", "OpenAI WIT-400M"),
    # OpenAI's ViT-B/32 (TorchScript archive): the same corpus and training recipe as the ViT-L/14 with a smaller
    # image encoder of the same family, and the same architecture as the two OpenCLIP ViT-B/32 models.
    "vitb32_openai": ("ViT-B-32", "openai_vitb32/ViT-B-32.pt", "OpenAI WIT-400M"),
}
OPENAI_RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
OPENAI_VITL14_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
OPENAI_VITB32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
XLMR_DIR = os.path.join(ROOT, "weights", "xlm_roberta_base")   # config.json and tokenizer.json of xlm-roberta-base
XLMR_SHA256 = "a5412f19cca70db8d434f68361f7254bd322a553455a63dbb709bd6a650cf378"   # open_clip_pytorch_model.bin


def create_openclip(key):
    """Model, image preprocessing, and tokenizer for one of the OpenCLIP checkpoints in weights/."""
    import copy
    import json
    import tempfile
    import open_clip
    arch, ckpt, _ = MODELS[key]
    path = os.path.join(ROOT, "weights", ckpt)
    kwargs = {}
    if arch.startswith("xlm-roberta"):
        cfg = copy.deepcopy(open_clip.get_model_config(arch))
        cfg["text_cfg"]["hf_model_name"] = cfg["text_cfg"]["hf_tokenizer_name"] = XLMR_DIR
        arch += "-local"
        cfg_dir = tempfile.mkdtemp()
        with open(os.path.join(cfg_dir, arch + ".json"), "w") as fh:
            json.dump(cfg, fh)
        open_clip.add_model_config(cfg_dir)
        kwargs["pretrained_text"] = False     # every weight comes from the checkpoint
    model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=path, **kwargs)
    return model, pre, open_clip.get_tokenizer(arch)


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
