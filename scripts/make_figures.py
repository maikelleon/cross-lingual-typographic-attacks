"""Draw the five manuscript figures in one shared style (vector PDF, plus PNG previews).

Palette: one color per word type, used the same way in every figure: pseudoword #2a78d6, translation #eb6834,
false friend #a23b9f, and English class name #199e70. The four were checked together for all pairs on white:
separation under simulated color-vision deficiency, separation under normal vision, and contrast of at least
3:1. Marker shapes add a second channel (circle, square, triangle, diamond), and text, axes, and reference lines
are black. The style follows the journal's figure guide: one serif face (STIX, bundled with matplotlib), no
shading, grid lines, or minor ticks, axes that end near the range of the data, lowercase bold panel labels in
parentheses at the top left of each panel, solid symbols of 7.5 pt, and 1.5 pt data lines. Every figure is
drawn on a canvas exactly as wide as the text (6 in), so LaTeX places it without rescaling and type sizes and
line weights are the same in every figure; the model names in Figures 3 and 5 share one label column.
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

PAPER = os.path.join(PROJECT, "outputs")
FIG = os.path.join(PAPER, "figures")
os.makedirs(FIG, exist_ok=True)
facts = json.load(open(os.path.join(PAPER, "data", "facts.json")))

INK = "#000000"
C = {"pseudoword": "#2a78d6", "translation": "#eb6834", "false_friend": "#a23b9f", "english": "#199e70"}
MK = {"pseudoword": "o", "translation": "s", "false_friend": "^", "english": "D"}
NAME = {"pseudoword": "Pseudoword", "translation": "Translation", "false_friend": "False friend"}
MS = 7.5                    # marker size (points) of every data symbol; scatter sizes are MS ** 2
DIAMOND = 0.9               # diamonds are drawn slightly smaller, so that they look as large as the other symbols
LW = 1.5                    # width (points) of every data line and error bar
ORDER = ["vitb32_laion400m", "vitb16_laion400m", "rn50_openai", "vitb32_openai", "vitl14_openai", "vitb32_xlmr_laion5b",
         "rn50_yfcc15m", "rn101_yfcc15m", "rn50_cc12m"]
READERS = ORDER[:6]
LABEL = {k: f"{facts['models'][k]['arch']}, {facts['models'][k]['corpus']}" for k in ORDER}
WIDTH = 6.0                 # inches, the text width of the manuscript
ROW = 0.3                   # inches per model in Figures 3 and 5

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.columnspacing": 1.8, "legend.handletextpad": 0.5,
})


def text_width(text, size):
    """Width in inches of a string set in the figure font at the given size."""
    fig = plt.figure(figsize=(1, 1))
    t = fig.text(0, 0, text, fontsize=size)
    width = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    plt.close(fig)
    return width


# The model names of Figures 3 and 5 sit in a label column of one width, so the two plots start at the same place.
LABEL_COL = max(text_width(LABEL[k], 7.5) for k in ORDER) + 2.5 / 72 + 0.02


def panel_label(fig, x, y, letter):
    """Bold panel label with its left edge at (x, y) in inches from the bottom left of the canvas."""
    w, h = fig.get_size_inches()
    fig.text(x / w, y / h, f"({letter})", fontsize=9, fontweight="bold", ha="left", va="baseline", color=INK)


def symbol(fig, x, y, wt, size=MS):
    """Word-type symbol centered at (x, y) in inches, with a thin white edge as in the plots."""
    w, h = fig.get_size_inches()
    fig.add_artist(Line2D([x / w], [y / h], transform=fig.transFigure, linestyle="none", marker=MK[wt],
                          markersize=size * (DIAMOND if MK[wt] == "D" else 1.0), color=C[wt], markeredgecolor="white",
                          markeredgewidth=1.0))


def save(fig, name, dpi=400):
    """Save the whole canvas, without cropping, after checking that nothing extends beyond it. No creation date in
    the PDF, so that a rerun reproduces the file byte for byte; images inside a PDF are resampled at the given dpi."""
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    margins = {"left": box.x0, "bottom": box.y0, "right": w - box.x1, "top": h - box.y1}
    assert all(v > -0.005 for v in margins.values()), (name, margins)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), dpi=dpi, metadata={"CreationDate": None})
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=400)
    plt.close(fig)
    return margins


def model_rows(ax):
    """Model names in the shared label column, one row per model, top to bottom in ORDER."""
    y = np.arange(len(ORDER))[::-1].astype(float)
    ax.set_yticks(y, [LABEL[k] for k in ORDER])
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.spines["left"].set_visible(False)
    return y


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
    # Inside every box the number sits in a narrow gutter on the left, and the title and the body text start at
    # the same indent; the symbols of the first box sit in the gutter, under the number.
    NUM, TEXT, PAD = 0.09, 0.27, 0.06

    def box(x, y, w, h, num, title):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="white", edgecolor=INK, linewidth=0.8))
        ax.text(x + NUM, y + h - 0.09, num, fontsize=9, fontweight="bold", color=INK, va="top", ha="left")
        ax.text(x + TEXT, y + h - 0.09, title, fontsize=8.5, fontweight="bold", color=INK, va="top", ha="left")

    def lines(x, y, items, right, dy=0.175):
        """Each item is a string or a list of (text, italic) pieces; example words are set in italics."""
        for i, item in enumerate(items):
            xx = x
            for txt, italic in (item if isinstance(item, list) else [(item, False)]):
                t = ax.text(xx, y - i * dy, txt, fontsize=8, color=INK, va="top", ha="left",
                            fontstyle="italic" if italic else "normal")
                xx += t.get_window_extent(renderer=renderer).width / fig.dpi
            assert xx <= right, (item, xx, right)           # every line ends inside its box

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color=INK,
                                     shrinkA=0, shrinkB=0))

    m, gap = 0.02, 0.14
    top, h1 = H - 0.02, 1.42
    w1, w2 = 2.50, 1.62
    w3 = W - 2 * m - w1 - w2 - 2 * gap
    x1, x2 = m, m + w1 + gap
    x3 = x2 + w2 + gap
    body = top - 0.34
    box(x1, top - h1, w1, h1, "1", f"Word sets, {facts['stimuli']['targets']} target classes")
    lines(x1 + TEXT, body, [
        [("English class name: ", False), ("lemon", True)],
        [("Translations: ", False), ("limón, citron, limão", True)],
        [("False friends: ", False), ("alemán, leçon", True)],
        [("Pseudowords: ", False), ("memon, legun, jamet", True)],
        [("Anagram, random string: ", False), ("eolmn, wwkvg", True)],
        "Spelling similarity to the name, 0 to 1",
    ], x1 + w1 - PAD)
    for i, wt in [(0, "english"), (1, "translation"), (2, "false_friend"), (3, "pseudoword")]:
        symbol(fig, x1 + NUM + 0.045, body - i * 0.175 - 0.058, wt)
    box(x2, top - h1, w2, h1, "2", "Lexical review")
    lines(x2 + TEXT, body, ["Native-speaker review", "of the Spanish words", "Same criteria for the",
                            "other four languages", "Rejected words dropped,", "replacements added"], x2 + w2 - PAD)
    box(x3, top - h1, w3, h1, "3", "Rendering")
    lines(x3 + TEXT, body, ["One word per image,", "white box, DejaVu Sans", "Photographs of other",
                            "classes, shared by all", "words of a target",
                            f"{facts['stimuli']['images_rendered']:,} stimulus images"], x3 + w3 - PAD)
    y2, h2 = 0.02, 1.42
    v1, v2 = 1.95, 1.62
    v3 = W - 2 * m - v1 - v2 - 2 * gap
    u1, u2 = m, m + v1 + gap
    u3 = u2 + v2 + gap
    body2 = y2 + h2 - 0.34
    box(u1, y2, v1, h2, "4", "Zero-shot classification")
    lines(u1 + TEXT, body2, ["Nine CLIP models in three", "controlled comparisons: data,", "caption language, and size",
                             "1,000 ImageNet classes", "Prompt: a photo of a {class}.",
                             f"{facts['stimuli']['run_total']:,} classifications"], u1 + v1 - PAD)
    box(u2, y2, v2, h2, "5", "Outcome")
    lines(u2 + TEXT, body2, ["Change in log $P$(target)", "relative to a random", "string on the same",
                             "photograph (nats)"], u2 + v2 - PAD)
    box(u3, y2, v3, h2, "6", "Analyses")
    lines(u3 + TEXT, body2, ["Reading index", "Spelling-similarity gradient", "Regressions controlling for",
                             "spelling, letter position,", "and text-box size", "Robustness checks"], u3 + v3 - PAD)
    arrow((x1 + w1, top - h1 / 2), (x2, top - h1 / 2))
    arrow((x2 + w2, top - h1 / 2), (x3, top - h1 / 2))
    ymid = (top - h1 + y2 + h2) / 2
    xa, xb = x3 + w3 / 2, u1 + v1 / 2
    ax.plot([xa, xa, xb], [top - h1, ymid, ymid], color=INK, linewidth=1.0, solid_joinstyle="round")
    arrow((xb, ymid), (xb, y2 + h2))
    arrow((u1 + v1, y2 + h2 / 2), (u2, y2 + h2 / 2))
    arrow((u2 + v2, y2 + h2 / 2), (u3, y2 + h2 / 2))
    return save(fig, "fig1_method")


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
    kinds = ["english", "translation", "translation", "false_friend", "pseudoword", None]
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

    def p3(v):
        pct = 100 * v
        if pct < 0.001:
            return "<0.001%"
        return f"{pct:.1f}%" if pct >= 10 else f"{pct:.2g}%"

    # Three square tiles per row, flush with both edges of the text; the title above each tile and the two
    # probabilities below it start at the tile's left edge, and every title starts at the same indent.
    tile = 1.86
    gap = (WIDTH - 3 * tile) / 2
    head, foot, between, line = 0.24, 0.40, 0.16, 0.165
    H = 2 * (head + tile + foot) + between
    fig = plt.figure(figsize=(WIDTH, H))
    for i, ((kind, word), img, wt) in enumerate(zip(words, imgs, kinds)):
        r, c = divmod(i, 3)
        x0 = c * (tile + gap)
        y_top = H - r * (head + tile + foot + between)
        y_img = y_top - head - tile
        ax = fig.add_axes([x0 / WIDTH, y_img / H, tile / WIDTH, tile / H])
        ax.imshow(img, extent=(0, 1, 0, 1), aspect="auto", interpolation="nearest")
        ax.set_axis_off()
        base = y_img + 0.075                 # baseline of the title, just above the tile
        panel_label(fig, x0, base + tile, chr(97 + i))
        if wt:
            symbol(fig, x0 + 0.31, base + tile + 0.04, wt)
        fig.text((x0 + 0.42) / WIDTH, (base + tile) / H, kind, fontsize=8, color=INK, ha="left", va="baseline")
        for j, (name, pred) in enumerate([("RN50, OpenAI WIT", reader), ("RN50, YFCC15M", nonreader)]):
            fig.text(x0 / WIDTH, (y_img - 0.16 - j * line) / H, f"{name}: {p3(pred[i][2])}", fontsize=8, color=INK,
                     ha="left", va="baseline")
    # Each pixel of the 224-pixel stimuli becomes exactly 4 x 4 pixels in the PDF, so the tiles show the images at
    # the resolution the models received, without smoothing.
    save(fig, "fig2_stimuli", dpi=4 * IMG_PX / tile)
    return reader, nonreader


# ---------------------------------------------------------------- Figure 3: reading strength
def figure3():
    d = pd.read_csv(os.path.join(PAPER, "data", "fig3_reading.csv")).set_index("model").loc[ORDER]
    H = 0.62 + ROW * len(ORDER)
    fig = plt.figure(figsize=(WIDTH, H))
    left, right, gap = LABEL_COL / WIDTH, 0.985, 0.06
    bottom, top = 0.40 / H, 1 - 0.22 / H
    span = right - left - gap
    a = fig.add_axes([left, bottom, span * 1.05 / 2.05, top - bottom])
    b = fig.add_axes([left + span * 1.05 / 2.05 + gap, bottom, span / 2.05, top - bottom])
    y = model_rows(a)
    xmax = 10 * int(np.ceil(100 * d.english_success.max() / 10))       # the axis ends at the decade above the data
    a.set_xlim(0, xmax)
    a.set_xticks(range(0, xmax + 1, 10))
    for yi, v in zip(y, d.english_success * 100):
        if v >= 0.5:                     # bars below 0.5% would be invisible slivers; their values are printed
            a.barh(yi, v, height=0.52, color=C["english"], linewidth=0)
        a.text(v + 1.0, yi, f"{v:.1f}%" if v >= 1 else ("0%" if v == 0 else f"{v:.2f}%"), va="center", ha="left",
               fontsize=7.5, color=INK, clip_on=False)
    a.set_xlabel("Predictions switched to the target class (%)")

    b.errorbar(d.reading_index, y, xerr=[d.reading_index - d.ri_lo, d.ri_hi - d.reading_index], fmt="none",
               ecolor=C["english"], elinewidth=LW, capsize=0)
    b.scatter(d.reading_index, y, s=(DIAMOND * MS) ** 2, marker=MK["english"], color=C["english"],
              edgecolor="white", linewidth=1.0, zorder=3, clip_on=False)
    for yi, v in zip(y, d.reading_index):
        b.text(v + 0.55, yi + 0.18, (f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}").replace("-", "−"), va="bottom",
               ha="left", fontsize=7.5, color=INK, clip_on=False)
    b.axvline(0, color=INK, linewidth=0.6, zorder=1)
    xmax_b = 5 * int(np.ceil((d.ri_hi.max() + 1.5) / 5))     # the next multiple of 5, with room for the values
    b.set_xlim(-0.5, xmax_b)
    b.set_xticks(range(0, xmax_b + 1, 5))
    b.set_ylim(a.get_ylim())
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.set_xlabel("Reading index (nats)")
    # Panel labels share one baseline at the top; (a) starts at the left edge of the label column.
    base = H - 0.11
    panel_label(fig, 0.0, base, "a")
    panel_label(fig, b.get_position().x0 * WIDTH - 0.02, base, "b")
    return save(fig, "fig3_reading")


# ---------------------------------------------------------------- Figure 4: similarity bands
def figure4():
    d = pd.read_csv(os.path.join(PAPER, "data", "fig4_similarity.csv"))
    bands = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    # Up to four reading models fill a 2 x 2 grid and five or six a 2 x 3 grid; the legend sits above the panels,
    # centered on the grid. Each panel has its own y scale, ending at the first multiple of 5 above its largest
    # value, so that no axis runs far beyond the data it shows.
    cols = 3 if len(READERS) > 4 else 2
    H = 4.4
    left, right, bottom, top = 0.50 / WIDTH, 0.99, 0.42 / H, 1 - 0.50 / H
    fig, axes = plt.subplots(2, cols, figsize=(WIDTH, H),
                             gridspec_kw={"wspace": 0.25, "hspace": 0.50, "left": left, "right": right,
                                          "bottom": bottom, "top": top})
    axes = axes.ravel()
    for ax, key in zip(axes, READERS):
        g = d[d.model == key]
        mids = np.array([0.1, 0.3, 0.5, 0.7, 0.9])           # band midpoints on a continuous similarity axis
        for wt in ["pseudoword", "translation", "false_friend"]:
            s = g[g.word_type == wt].set_index("band").reindex(bands)
            ok = s["mean"].notna().values
            ax.plot(mids[ok], 100 * s["mean"].values[ok], color=C[wt], linewidth=LW, solid_capstyle="round",
                    solid_joinstyle="round", marker=MK[wt], markersize=MS, markeredgecolor="white",
                    markeredgewidth=1.0, label=NAME[wt], zorder=3, clip_on=False)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ymax = 5 * int(np.ceil(100 * g["mean"].max() / 5))
        ax.set_ylim(0, ymax)
        ax.set_yticks(range(0, ymax + 1, 5))
        ax.set_xlim(0, 1)
    # Each panel label starts at the left edge of its panel's tick labels, and the model name, set like the model
    # names of Figures 3 and 5, follows on the same baseline just above the panel.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, key, letter in zip(axes, READERS, "abcdef"):
        pos = ax.get_position()
        edge = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels()) / fig.dpi
        base = pos.y1 * H + 0.07
        panel_label(fig, edge, base, letter)
        fig.text(pos.x0 + 0.03 / WIDTH, base / H, LABEL[key], fontsize=7.5, ha="left", va="baseline", color=INK)
    for ax in axes[len(READERS):]:
        ax.axis("off")
    fig.supylabel("Predictions switched to the target class (%)", fontsize=8, x=0.02 / WIDTH, ha="left")
    fig.supxlabel("Spelling similarity to the English class name", fontsize=8, x=(left + right) / 2,
                  y=0.04 / H, va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=((left + right) / 2, 1 - 0.02 / H))
    return save(fig, "fig4_similarity")


# ---------------------------------------------------------------- Figure 5: adjusted effects
def figure5():
    d = pd.read_csv(os.path.join(PAPER, "data", "fig5_effects.csv")).set_index("model").loc[ORDER]
    H = 0.85 + ROW * len(ORDER)
    fig = plt.figure(figsize=(WIDTH, H))
    left, right, bottom, top = LABEL_COL / WIDTH, 0.985, 0.40 / H, 1 - 0.36 / H
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    y = model_rows(ax)
    for col, wt, off in [("tr", "translation", 0.14), ("ff", "false_friend", -0.14)]:
        est, se = d[col].values, d[f"{col}_se"].values
        ax.errorbar(est, y + off, xerr=1.96 * se, fmt="none", ecolor=C[wt], elinewidth=LW, capsize=0, zorder=2)
        ax.scatter(est, y + off, s=MS ** 2, marker=MK[wt], color=C[wt], edgecolor="white", linewidth=1.0, zorder=3,
                   label=f"{NAME[wt]} vs. pseudoword")
    ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
    ax.axhline(len(ORDER) - len(READERS) - 0.5, color=INK, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    lo = min((d[c] - 1.96 * d[f"{c}_se"]).min() for c in ("tr", "ff"))
    hi = max((d[c] + 1.96 * d[f"{c}_se"]).max() for c in ("tr", "ff"))
    xmax = np.ceil(hi * 10) / 10 + 0.05
    ax.set_xlim(np.floor(lo * 10) / 10 - 0.05, xmax)   # just beyond the intervals
    ax.set_xticks(np.arange(np.floor(lo * 2) / 2, xmax + 1e-9, 0.5))
    ax.set_xlabel("Difference in log-probability of the target class (nats)")
    # As in Figure 4, the legend sits above the plot, centered on it.
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=((left + right) / 2,
                                                                                            1 - 0.02 / H))
    return save(fig, "fig5_effects")


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
