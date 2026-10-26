"""Draw the five manuscript figures in one shared style (vector PDF, plus PNG previews).

Palette: categorical slots validated for all pairs on white, with no red-green pairing: pseudoword #2a78d6,
translation #eb6834, false friend #a23b9f, and black for the English class name; marker shapes add a second
channel (circle, square, triangle, diamond). The style follows the journal's figure guide: one serif face
(STIX, bundled with matplotlib), black type, axes, and reference lines, no shading, grid lines, or minor
ticks, axes that end near the range of the data, lowercase panel labels in parentheses, solid symbols of at
least 6 pt, and 1.5 pt data lines.
"""
import json
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(PROJECT, "code")
sys.path.insert(0, CODE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

PAPER = os.path.join(PROJECT, "outputs")
FIG = os.path.join(PAPER, "figures")
os.makedirs(FIG, exist_ok=True)
facts = json.load(open(os.path.join(PAPER, "data", "facts.json")))

INK = "#000000"
C = {"pseudoword": "#2a78d6", "translation": "#eb6834", "false_friend": "#a23b9f", "english": INK}
MK = {"pseudoword": "o", "translation": "s", "false_friend": "^", "english": "D"}
NAME = {"pseudoword": "Pseudoword", "translation": "Translation", "false_friend": "False friend"}
ORDER = ["vitb32_laion400m", "vitb16_laion400m", "rn50_openai", "rn50_yfcc15m", "rn101_yfcc15m", "rn50_cc12m"]
LABEL = {k: f"{facts['models'][k]['arch']}, {facts['models'][k]['corpus']}" for k in ORDER}
WIDTH = 6.0  # inches, the text width of the manuscript

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    "legend.frameon": False,
})


def panel_label(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right",
            color=INK)


def save(fig, name):
    # No creation date in the PDF, so that a rerun reproduces the file byte for byte.
    # Photographs inside a PDF are resampled at 400 dpi, which keeps them above 300 dpi after LaTeX scales
    # the figure to the text width.
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02, dpi=400,
                metadata={"CreationDate": None})
    fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)


def hbar(ax, y, value, height, color, r=0.06):
    """Horizontal bar with a rounded data end and a square baseline."""
    if value <= 0:
        return
    rr = min(r * (ax.get_xlim()[1] - ax.get_xlim()[0]) / 6, value / 2)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), value, height, boxstyle=f"round,pad=0,rounding_size={rr}",
                                mutation_aspect=height / rr if rr else 1, linewidth=0, facecolor=color))
    ax.add_patch(Rectangle((0, y - height / 2), min(value, rr * 1.5), height, linewidth=0, facecolor=color))



# ---------------------------------------------------------------- Figure 1: experimental design
def figure1():
    from matplotlib.patches import FancyArrowPatch
    W, H = WIDTH, 3.2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    renderer = fig.canvas.get_renderer()

    def box(x, y, w, h, num, title):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="white", edgecolor=INK, linewidth=0.8))
        ax.text(x + 0.09, y + h - 0.085, num, fontsize=9.5, fontweight="bold", color=INK, va="top", ha="left")
        ax.text(x + 0.26, y + h - 0.09, title, fontsize=8.5, fontweight="bold", color=INK, va="top", ha="left")

    def lines(x, y, items, dy=0.175):
        """Each item is a string or a list of (text, italic) pieces; example words are set in italics."""
        for i, item in enumerate(items):
            xx = x
            for txt, italic in (item if isinstance(item, list) else [(item, False)]):
                t = ax.text(xx, y - i * dy, txt, fontsize=8, color=INK, va="top", ha="left",
                            fontstyle="italic" if italic else "normal")
                xx += t.get_window_extent(renderer=renderer).width / fig.dpi

    def marker(x, y, wt):
        # The white edge is drawn half inside the marker, so these sizes leave at least 6 pt of solid color.
        ax.plot([x], [y], marker=MK[wt], markersize=7.5 if wt != "english" else 6.8, color=C[wt],
                markeredgecolor="white", markeredgewidth=0.6)

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color=INK,
                                     shrinkA=0, shrinkB=0))

    m, gap = 0.02, 0.14
    top, h1 = H - 0.02, 1.42
    w1, w2 = 2.50, 1.62
    w3 = W - 2 * m - w1 - w2 - 2 * gap
    x1, x2 = m, m + w1 + gap
    x3 = x2 + w2 + gap
    box(x1, top - h1, w1, h1, "1", f"Word sets, {facts['stimuli']['targets']} target classes")
    lines(x1 + 0.27, top - 0.34, [
        [("English class name: ", False), ("lemon", True)],
        [("Translations: ", False), ("limón, citron, limão", True)],
        [("False friends: ", False), ("alemán, leçon", True)],
        [("Pseudowords: ", False), ("memon, legun, jamet", True)],
        [("Anagram, random string: ", False), ("eolmn, wwkvg", True)],
        "Spelling similarity to the name, 0 to 1",
    ])
    for i, wt in [(0, "english"), (1, "translation"), (2, "false_friend"), (3, "pseudoword")]:
        marker(x1 + 0.17, top - 0.34 - i * 0.175 - 0.058, wt)
    box(x2, top - h1, w2, h1, "2", "Lexical review")
    lines(x2 + 0.11, top - 0.34, ["Native-speaker review", "of the Spanish words", "Same criteria for the",
                                  "other four languages", "Rejected words dropped,", "replacements added"])
    box(x3, top - h1, w3, h1, "3", "Rendering")
    lines(x3 + 0.11, top - 0.34, ["One word per image,", "white box, DejaVu Sans", "Photographs of other",
                                  "classes, shared by all", "words of a target",
                                  f"{facts['stimuli']['images_rendered']:,} stimulus images"])
    y2, h2 = 0.02, 1.42
    v1, v2 = 1.95, 1.62
    v3 = W - 2 * m - v1 - v2 - 2 * gap
    u1, u2 = m, m + v1 + gap
    u3 = u2 + v2 + gap
    box(u1, y2, v1, h2, "4", "Zero-shot classification")
    lines(u1 + 0.11, y2 + h2 - 0.34, ["Six CLIP models; three share", "an image-encoder architecture",
                                      "but differ in training", "1,000 ImageNet classes", "Prompt: a photo of a {class}.",
                                      f"{facts['stimuli']['run_total']:,} classifications"])
    box(u2, y2, v2, h2, "5", "Outcome")
    lines(u2 + 0.11, y2 + h2 - 0.34, ["Change in log $P$(target)", "relative to a random", "string on the same",
                                      "photograph (nats)"])
    box(u3, y2, v3, h2, "6", "Analyses")
    lines(u3 + 0.11, y2 + h2 - 0.34, ["Reading index", "Spelling-similarity gradient", "Regressions controlling for",
                                      "spelling, letter position,", "and text-box size", "Robustness checks"])
    arrow((x1 + w1, top - h1 / 2), (x2, top - h1 / 2))
    arrow((x2 + w2, top - h1 / 2), (x3, top - h1 / 2))
    ymid = (top - h1 + y2 + h2) / 2
    xa, xb = x3 + w3 / 2, u1 + v1 / 2
    ax.plot([xa, xa, xb], [top - h1, ymid, ymid], color=INK, linewidth=1.0, solid_joinstyle="round")
    arrow((xb, ymid), (xb, y2 + h2))
    arrow((u1 + v1, y2 + h2 / 2), (u2, y2 + h2 / 2))
    arrow((u2 + v2, y2 + h2 / 2), (u3, y2 + h2 / 2))
    save(fig, "fig1_method")


# ---------------------------------------------------------------- Figure 2: example stimuli
# The photograph is a public-domain work of the Federal Bureau of Investigation (Wikimedia Commons file
# "FBI Police K9 (33512795453).jpg"), described in photos/README.md. It is not one of the stimulus photographs.
PHOTO = "fbi_police_k9_33512795453.jpg"
CROP = (0.239, 0.029, 0.699)  # square around the dog: left and top as fractions of width and height, side of height


def figure2(recompute=False):
    from PIL import Image
    from common import IMG_PX, MODELS, ROOT, overlay

    words = [("English class name", "lemon"), ("Translation, French", "citron"), ("Translation, Spanish", "limón"),
             ("False friend, French", "leçon"), ("Pseudoword", "memon"), ("Random string", "wwkvg")]
    im = Image.open(os.path.join(PROJECT, "photos", PHOTO)).convert("RGB")
    w, h = im.size
    left, top, side = round(CROP[0] * w), round(CROP[1] * h), round(CROP[2] * h)
    photo = im.crop((left, top, left + side, top + side)).resize((IMG_PX, IMG_PX))
    imgs = [overlay(photo, w) for _, w in words]

    def predict(key):
        import torch
        import open_clip
        from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES

        torch.set_num_threads(2)
        arch, ckpt, _ = MODELS[key]
        path = os.path.join(ROOT, "weights", ckpt)
        if key == "rn50_openai":
            model = open_clip.openai.load_openai_model(path, precision="fp32", device="cpu")
            pre = open_clip.image_transform(model.visual.image_size, is_train=False)
        else:
            model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=path)
        model.eval()
        tok = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            t = model.encode_text(tok([f"a photo of a {n}." for n in IMAGENET_CLASSNAMES]))
            t = t / t.norm(dim=-1, keepdim=True)
            f = model.encode_image(torch.stack([pre(i) for i in imgs]))
            f = f / f.norm(dim=-1, keepdim=True)
            p = torch.softmax(model.logit_scale.exp() * f @ t.T, dim=-1)
        top = p.max(dim=-1)
        target = IMAGENET_CLASSNAMES.index("lemon")
        return [(IMAGENET_CLASSNAMES[i], float(v), float(pt)) for v, i, pt in zip(top.values, top.indices, p[:, target])]

    # Predictions are cached in data/fig2_predictions.json; pass --recompute to rerun both models.
    pred_path = os.path.join(PAPER, "data", "fig2_predictions.json")
    saved = json.load(open(pred_path, encoding="utf-8")) if os.path.exists(pred_path) else None
    stale = saved is None or saved.get("photo") != PHOTO or saved.get("crop") != list(CROP) or \
        [w for _, w in saved["words"]] != [w for _, w in words]
    if recompute or stale:
        reader, nonreader = predict("rn50_openai"), predict("rn50_yfcc15m")
        json.dump({"photo": PHOTO, "crop": list(CROP), "words": words, "rn50_openai": reader,
                   "rn50_yfcc15m": nonreader}, open(pred_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    else:
        reader, nonreader = saved["rn50_openai"], saved["rn50_yfcc15m"]

    fig = plt.figure(figsize=(WIDTH, 5.0))
    tile, gapx, top0, rowh = 0.30, 0.05, 0.98, 0.49
    for i, ((kind, word), img) in enumerate(zip(words, imgs)):
        r, c = divmod(i, 3)
        x0 = 0.025 + c * (tile + gapx)
        ax = fig.add_axes([x0, top0 - r * rowh - 0.335, tile, 0.335])
        ax.imshow(img)
        ax.set_axis_off()
        ax.text(0, 1.03, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left",
                va="bottom")
        ax.text(0.12, 1.03, kind, transform=ax.transAxes, fontsize=8, color=INK, ha="left", va="bottom")
        ytxt = top0 - r * rowh - 0.335 - 0.02

        def p3(v):
            pct = 100 * v
            if pct < 0.001:
                return "<0.001%"
            return f"{pct:.1f}%" if pct >= 10 else f"{pct:.2g}%"

        fig.text(x0, ytxt, f"RN50, OpenAI WIT: {p3(reader[i][2])}", fontsize=8, color=INK, va="top")
        fig.text(x0, ytxt - 0.035, f"RN50, YFCC15M: {p3(nonreader[i][2])}", fontsize=8, color=INK, va="top")
    save(fig, "fig2_stimuli")
    return reader, nonreader


# ---------------------------------------------------------------- Figure 3: reading strength
def figure3():
    d = pd.read_csv(os.path.join(PAPER, "data", "fig3_reading.csv")).set_index("model").loc[ORDER]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.35), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.16})
    b.patch.set_alpha(0)
    y = np.arange(len(ORDER))[::-1]
    a.set_xlim(0, 50)                    # the largest value is 49.1%
    a.set_xticks(range(0, 51, 10))
    for yi, v in zip(y, d.english_success * 100):
        if v >= 0.5:                     # bars below 0.5% would be invisible slivers; their values are printed
            hbar(a, yi, v, 0.52, C["english"])
        a.text(v + 1.0, yi, f"{v:.1f}%" if v >= 1 else ("0%" if v == 0 else f"{v:.2f}%"), va="center", ha="left",
               fontsize=7.5, color=INK,
               clip_on=False)
    a.set_yticks(y, [LABEL[k] for k in ORDER])
    a.set_ylim(-0.6, len(ORDER) - 0.4)
    a.set_xlabel("Predictions switched to the target class (%)")
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    panel_label(a, "a", x=-0.58)

    b.errorbar(d.reading_index, y, xerr=[d.reading_index - d.ri_lo, d.ri_hi - d.reading_index], fmt="none",
               ecolor=C["english"], elinewidth=1.5, capsize=0)
    b.scatter(d.reading_index, y, s=56, marker=MK["english"], color=C["english"], edgecolor="white", linewidth=1.0,
              zorder=3, clip_on=False)
    for yi, v in zip(y, d.reading_index):
        b.text(v + 0.55, yi + 0.18, (f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}").replace("-", "−"), va="bottom",
               ha="left", fontsize=7.5, color=INK, clip_on=False)
    b.axvline(0, color=INK, linewidth=0.6, zorder=1)
    b.set_xlim(-0.5, 13.5)               # the data run from -0.20 to 13.1 nats
    b.set_xticks([0, 5, 10])
    b.set_ylim(-0.6, len(ORDER) - 0.4)
    b.set_yticks(y, [""] * len(ORDER))
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.set_xlabel("Reading index (nats)")
    panel_label(b, "b", x=-0.02)
    save(fig, "fig3_reading")


# ---------------------------------------------------------------- Figure 4: similarity bands
def figure4():
    d = pd.read_csv(os.path.join(PAPER, "data", "fig4_similarity.csv"))
    bands = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    readers = ORDER[:3]
    # Each panel has its own y scale, ending at the first multiple of 5 above its largest value, so that no axis
    # runs far beyond the data it shows.
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), gridspec_kw={"wspace": 0.3})
    for ax, key, letter in zip(axes, readers, "abc"):
        g = d[d.model == key]
        mids = np.array([0.1, 0.3, 0.5, 0.7, 0.9])           # band midpoints on a continuous similarity axis
        for wt in ["pseudoword", "translation", "false_friend"]:
            s = g[g.word_type == wt].set_index("band").reindex(bands)
            ok = s["mean"].notna().values
            ax.plot(mids[ok], 100 * s["mean"].values[ok], color=C[wt], linewidth=1.5, solid_capstyle="round",
                    solid_joinstyle="round", marker=MK[wt], markersize=7.5, markeredgecolor="white",
                    markeredgewidth=1.0, label=NAME[wt], zorder=3, clip_on=False)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.set_title(LABEL[key], loc="left", pad=4)
        top = 5 * int(np.ceil(100 * g["mean"].max() / 5))
        ax.set_ylim(0, top)
        ax.set_yticks(range(0, top + 1, 5))
        ax.set_xlim(0, 1)
        panel_label(ax, letter, x=-0.03, y=1.02)
    axes[0].set_ylabel("Predictions switched to the target class (%)")
    fig.supxlabel("Spelling similarity to the English class name", fontsize=8, y=-0.04)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), handlelength=2.2, columnspacing=1.8)
    save(fig, "fig4_similarity")


# ---------------------------------------------------------------- Figure 5: adjusted effects
def figure5():
    d = pd.read_csv(os.path.join(PAPER, "data", "fig5_effects.csv")).set_index("model").loc[ORDER]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    y = np.arange(len(ORDER))[::-1].astype(float)
    for col, wt, off in [("tr", "translation", 0.14), ("ff", "false_friend", -0.14)]:
        est, se = d[col].values, d[f"{col}_se"].values
        ax.errorbar(est, y + off, xerr=1.96 * se, fmt="none", ecolor=C[wt], elinewidth=1.5, capsize=0, zorder=2)
        ax.scatter(est, y + off, s=64, marker=MK[wt], color=C[wt], edgecolor="white", linewidth=1.0, zorder=3,
                   label=f"{NAME[wt]} vs. pseudoword")
    ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
    ax.axhline(2.5, color=INK, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    ax.set_yticks(y, [LABEL[k] for k in ORDER])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_xlim(-1.5, 0.6)                   # the intervals run from -1.42 to 0.53 nats
    ax.set_xticks([-1.5, -1.0, -0.5, 0, 0.5])
    ax.set_xlabel("Difference in log-probability of the target class (nats)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.4, columnspacing=1.8)
    save(fig, "fig5_effects")


if __name__ == "__main__":
    recompute = "--recompute" in sys.argv
    which = [a for a in sys.argv[1:] if a != "--recompute"] or ["1", "2", "3", "4", "5"]
    if "1" in which:
        figure1()
    if "2" in which:
        figure2(recompute)
    if "3" in which:
        figure3()
    if "4" in which:
        figure4()
    if "5" in which:
        figure5()
