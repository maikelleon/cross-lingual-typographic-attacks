"""Cross-model summary, including a position-aware control for the translation effect.

Cognates tend to keep the start and end of the English word, while pseudowords change letters at
random positions; the position-aware model adds prefix, suffix and letter-bigram overlap so that a
remaining translation advantage cannot be blamed on where the shared letters sit.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rapidfuzz.distance import JaroWinkler

from analyze import load
from common import MODELS, strip_accents

ORDER = ["vitb32_laion400m", "vitb16_laion400m", "rn50_openai", "rn50_yfcc15m", "rn101_yfcc15m", "rn50_cc12m",
         "vitb32_xlmr_laion5b", "vitl14_openai"]


def prefix_ratio(w, e):
    w, e = strip_accents(w.lower()), e.lower()
    n = 0
    for a, b in zip(w, e):
        if a != b:
            break
        n += 1
    return n / len(e)


def suffix_ratio(w, e):
    return prefix_ratio(w[::-1], e[::-1])


def dice(w, e):
    w, e = strip_accents(w.lower()), e.lower()
    bw = [w[i:i + 2] for i in range(len(w) - 1)]
    be = [e[i:i + 2] for i in range(len(e) - 1)]
    if not bw or not be:
        return 0.0
    common = sum(min(bw.count(b), be.count(b)) for b in set(bw))
    return 2 * common / (len(bw) + len(be))


def tr_vs_ff_matched(df):
    """Translations vs false friends inside the spelling range both share, controlling for the
    spelling profile, word frequency (English and own language) and length."""
    real = df[df.word_type.isin(["translation", "false_friend"]) & (df.sim_raw < 0.999)].dropna(subset=["zipf_lang"]).copy()
    real["zipf_lang"] = real["zipf_lang"].astype(float)
    real["prefix"] = [prefix_ratio(w, e) for w, e in zip(real.word, real.target_en)]
    real["suffix"] = [suffix_ratio(w, e) for w, e in zip(real.word, real.target_en)]
    real["dice"] = [dice(w, e) for w, e in zip(real.word, real.target_en)]
    real["jw"] = [JaroWinkler.normalized_similarity(strip_accents(w.lower()), e) for w, e in zip(real.word, real.target_en)]
    real["word_type"] = pd.Categorical(real["word_type"], ["false_friend", "translation"])
    lo, hi = real[real.word_type == "false_friend"].sim_raw.quantile([0.02, 0.98])
    band = real[(real.sim_raw >= lo) & (real.sim_raw <= hi)]
    f = ("d_target_logp ~ bs(sim_raw, df=4) + bs(jw, df=3) + prefix + suffix + dice + zipf_lang + zipf_en"
         " + n_chars + C(word_type)")
    m = smf.ols(f, band).fit(cov_type="cluster", cov_kwds={"groups": band["target_idx"]})
    k = "C(word_type)[T.translation]"
    return m.params[k], m.bse[k], m.pvalues[k], len(band)


def ff_position_effect(df):
    sub = df[df.word_type.isin(["pseudoword", "false_friend"]) & (df.sim_raw < 0.999)].copy()
    sub["prefix"] = [prefix_ratio(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["suffix"] = [suffix_ratio(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["dice"] = [dice(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["jw"] = [JaroWinkler.normalized_similarity(strip_accents(w.lower()), e) for w, e in zip(sub.word, sub.target_en)]
    sub["word_type"] = pd.Categorical(sub["word_type"], ["pseudoword", "false_friend"])
    m = smf.ols("d_target_logp ~ bs(sim_raw, df=4) + bs(jw, df=3) + prefix + suffix + dice + C(word_type)", sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["target_idx"]})
    k = "C(word_type)[T.false_friend]"
    return m.params[k], m.bse[k], m.pvalues[k]


_BOX = {}


def text_box(word):
    """Rendered text width and height in pixels (same font fitting as the stimuli)."""
    from PIL import Image, ImageDraw
    from common import fit_font, font
    if word not in _BOX:
        b = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox((0, 0), word, font=font(fit_font(word)))
        _BOX[word] = (b[2] - b[0], b[3] - b[1])
    return _BOX[word]


def box_adjusted(df):
    """Translation and false-friend effects vs pseudowords, adjusted for the spelling profile and for
    the rendered text box (width relative to the random string on the same photo, and height): a wider
    box hides more of the photo, which shifts every class a little even in models that do not read."""
    sub = df[df.word_type.isin(["pseudoword", "translation", "false_friend"]) & (df.sim_raw < 0.999)].copy()
    rnd = df[df.word_type == "random"].drop_duplicates(["target_idx"])[["target_idx", "word"]].rename(columns={"word": "rnd"})
    sub = sub.merge(rnd, on="target_idx", how="left")
    sub["box_h"] = [text_box(w)[1] for w in sub.word]
    sub["dw"] = [text_box(w)[0] - text_box(r)[0] for w, r in zip(sub.word, sub.rnd)]
    sub["prefix"] = [prefix_ratio(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["suffix"] = [suffix_ratio(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["dice"] = [dice(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["jw"] = [JaroWinkler.normalized_similarity(strip_accents(w.lower()), e) for w, e in zip(sub.word, sub.target_en)]
    sub["word_type"] = pd.Categorical(sub["word_type"], ["pseudoword", "translation", "false_friend"])
    f = "d_target_logp ~ bs(sim_raw, df=4) + bs(jw, df=3) + prefix + suffix + dice + bs(dw, df=3) + box_h + C(word_type)"
    m = smf.ols(f, sub).fit(cov_type="cluster", cov_kwds={"groups": sub["target_idx"]})
    out = {}
    for t, label in [("translation", "tr"), ("false_friend", "ff")]:
        k = f"C(word_type)[T.{t}]"
        out[f"{label}_box"], out[f"{label}_box_se"], out[f"{label}_box_p"] = m.params[k], m.bse[k], m.pvalues[k]
    return out


def summarize(key):
    df, clean = load(key)
    df = df.dropna(subset=["d_target_logp"])
    en = df[df.word_type == "english"].groupby("target_idx")["d_target_logp"].mean()
    rng = np.random.default_rng(0)
    boots = [en.sample(len(en), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 1000)]
    row = {
        "model": key, "corpus": MODELS[key][2], "arch": MODELS[key][0],
        "clean_top1": (clean.argmax(1) == np.arange(1000)).mean(),
        "reading_index": en.mean(), "ri_lo": np.percentile(boots, 2.5), "ri_hi": np.percentile(boots, 97.5),
        "english_success": df[df.word_type == "english"].hit.mean(),
        "noncognate_translation_success": df[(df.word_type == "translation") & (df.sim_raw < 0.4)].hit.mean(),
        "cognate_translation_success": df[(df.word_type == "translation") & (df.sim_raw >= 0.4) & (df.sim_raw < 0.999)].hit.mean(),
    }
    sub = df[df.word_type.isin(["translation", "pseudoword", "false_friend"]) & (df.sim_raw < 0.999)].copy()
    sub["prefix"] = [prefix_ratio(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["suffix"] = [suffix_ratio(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["dice"] = [dice(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["jw"] = [JaroWinkler.normalized_similarity(strip_accents(w.lower()), e) for w, e in zip(sub.word, sub.target_en)]
    sub["word_type"] = pd.Categorical(sub["word_type"], ["pseudoword", "translation", "false_friend"])
    specs = {
        "basic": "d_target_logp ~ bs(sim_raw, df=4) + C(word_type)",
        "position": "d_target_logp ~ bs(sim_raw, df=4) + bs(jw, df=3) + prefix + suffix + dice + C(word_type)",
    }
    for name, f in specs.items():
        m = smf.ols(f, sub).fit(cov_type="cluster", cov_kwds={"groups": sub["target_idx"]})
        for t, label in [("translation", "tr"), ("false_friend", "ff")]:
            k = f"C(word_type)[T.{t}]"
            row[f"{label}_{name}"] = m.params[k]
            row[f"{label}_{name}_se"] = m.bse[k]
            row[f"{label}_{name}_p"] = m.pvalues[k]
    pw = df[df.word_type == "pseudoword"]
    m1 = smf.ols("d_target_logp ~ sim_raw", pw).fit(cov_type="cluster", cov_kwds={"groups": pw["target_idx"]})
    row["pseudoword_slope"], row["pseudoword_slope_se"] = m1.params["sim_raw"], m1.bse["sim_raw"]
    row["tr_vs_ff"], row["tr_vs_ff_se"], row["tr_vs_ff_p"], row["tr_vs_ff_n"] = tr_vs_ff_matched(df)
    row.update(box_adjusted(df))
    # Sensitivity: false friends whose meaning names another ImageNet class removed.
    df2, _ = load(key, exclude_other_class=True)
    df2 = df2.dropna(subset=["d_target_logp"])
    row["ff_position_noother"], row["ff_position_noother_se"], row["ff_position_noother_p"] = ff_position_effect(df2)
    return row


if __name__ == "__main__":
    import os
    import sys
    RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    keys = sys.argv[1:] or [k for k in ORDER if os.path.exists(os.path.join(RESULTS, f"{k}_stimuli.csv"))]
    rows = [summarize(k) for k in keys]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "cross_model_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    show = out[["model", "corpus", "clean_top1", "reading_index", "ri_lo", "ri_hi", "english_success",
                "noncognate_translation_success", "cognate_translation_success", "pseudoword_slope",
                "tr_basic", "tr_basic_p", "tr_position", "tr_position_se", "tr_position_p",
                "ff_basic", "ff_position", "ff_position_se", "ff_position_p"]]
    print(show.round(4).to_string(index=False))
    print()
    print(out[["model", "tr_vs_ff", "tr_vs_ff_se", "tr_vs_ff_p", "tr_vs_ff_n",
               "ff_position_noother", "ff_position_noother_se", "ff_position_noother_p"]].round(4).to_string(index=False))
    print()
    print(out[["model", "tr_box", "tr_box_se", "tr_box_p", "ff_box", "ff_box_se", "ff_box_p"]].round(4).to_string(index=False))
