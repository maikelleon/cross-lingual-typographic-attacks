"""Compute every number, table and figure dataset used in the manuscript from the result files.

Outputs (outputs/): data/facts.json, data/fig3_reading.csv, data/fig4_similarity.csv, data/fig5_effects.csv,
tables/table_models.tex, tables/table_stimuli.tex. The table of effects, which combines these estimates with
the robustness checks, is written by robustness.py.
"""
import json
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(PROJECT, "code")
sys.path.insert(0, CODE)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analyze import load
from common import MODELS
from synthesis import box_adjusted, ff_position_effect, tr_vs_ff_matched, prefix_ratio, suffix_ratio, dice

PAPER = os.path.join(PROJECT, "outputs")
RES = os.path.join(PROJECT, "results")
ORDER = ["vitb32_laion400m", "vitb16_laion400m", "rn50_openai", "vitb32_openai", "vitl14_openai", "vitb32_xlmr_laion5b",
         "rn50_yfcc15m", "rn101_yfcc15m", "rn50_cc12m"]
READERS = ORDER[:6]
LABEL = {
    "vitb32_laion400m": ("ViT-B/32", "LAION-400M", 400), "vitb16_laion400m": ("ViT-B/16", "LAION-400M", 400),
    "rn50_openai": ("RN50", "OpenAI WIT", 400), "vitb32_xlmr_laion5b": ("ViT-B/32", "LAION-5B (multilingual)", 5850),
    "vitl14_openai": ("ViT-L/14", "OpenAI WIT", 400), "vitb32_openai": ("ViT-B/32", "OpenAI WIT", 400),
    "rn50_yfcc15m": ("RN50", "YFCC15M", 15), "rn101_yfcc15m": ("RN101", "YFCC15M", 15), "rn50_cc12m": ("RN50", "CC12M", 12),
}
PHOTOS = {"vitb32_laion400m": 16, "vitb32_xlmr_laion5b": 16}
# Parameters of each image encoder in millions, counted from the "visual." entries of the checkpoints.
IMAGE_PARAMS_M = {"vitb32_laion400m": 87.85, "vitb16_laion400m": 86.19, "rn50_openai": 38.37, "vitl14_openai": 303.97,
                  "vitb32_openai": 87.85,
                  "vitb32_xlmr_laion5b": 87.85, "rn50_yfcc15m": 38.37, "rn101_yfcc15m": 56.37, "rn50_cc12m": 38.37}
for sub in ("data", "tables"):
    os.makedirs(os.path.join(PAPER, sub), exist_ok=True)
summary = pd.read_csv(os.path.join(RES, "cross_model_summary.csv")).set_index("model")
facts = {"models": {}}


def ci(est, se):
    return [float(est - 1.96 * se), float(est + 1.96 * se)]


def fmt(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "\\textminus ") if s.startswith("-") else s


def est_ci(est, se, d=2):
    lo, hi = ci(est, se)
    return f"{fmt(est, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


bands = [0, 0.2, 0.4, 0.6, 0.8, 0.999]
band_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
fig3, fig4, table_rows = [], [], []
words_all = pd.read_csv(os.path.join(PROJECT, "stimuli", "words.csv"))
man_all = pd.read_csv(os.path.join(PROJECT, "stimuli", "manifest.csv"))

for key in ORDER:
    df, clean = load(key)
    df = df.dropna(subset=["d_target_logp"])
    s = summary.loc[key]
    m = {"arch": LABEL[key][0], "corpus": LABEL[key][1], "pairs_m": LABEL[key][2], "image_params_m": IMAGE_PARAMS_M[key],
         "photos_per_class": PHOTOS.get(key, 10), "stimuli_analyzed": int(len(df)),
         "stimuli_run": int(len(pd.read_csv(os.path.join(RES, f"{key}_stimuli.csv")))),
         "clean_top1": float((clean.argmax(1) == np.arange(1000)).mean()),
         "reading_index": float(s.reading_index), "ri_ci": [float(s.ri_lo), float(s.ri_hi)],
         "english_success": float(df[df.word_type == "english"].hit.mean()),
         "random_success": float(df[df.word_type == "random"].hit.mean()),
         "anagram_success": float(df[df.word_type == "anagram"].hit.mean())}
    tr = df[df.word_type == "translation"]
    m["noncognate_success"] = float(tr[tr.sim_raw < 0.4].hit.mean())
    m["noncognate_n"] = int((tr.sim_raw < 0.4).sum())
    m["cognate_success"] = float(tr[(tr.sim_raw >= 0.4) & (tr.sim_raw < 0.999)].hit.mean())
    m["identical_success"] = float(tr[tr.sim_raw >= 0.999].hit.mean())
    m["identical_n"] = int((tr.sim_raw >= 0.999).sum())
    m["translation_success_by_lang"] = {l: float(g.hit.mean()) for l, g in tr.groupby("lang")}
    m["translation_sim_by_lang"] = {l: float(g.sim_raw.mean()) for l, g in tr.groupby("lang")}
    # Accent sensitivity of the translation effect.
    a = smf.ols("d_target_logp ~ sim_raw", tr).fit()
    b = smf.ols("d_target_logp ~ sim_noaccent", tr).fit()
    m["r2_accent"], m["r2_noaccent"] = float(a.rsquared), float(b.rsquared)
    # The same comparison among the translations whose two similarity measures differ.
    trd = tr[(tr.sim_raw - tr.sim_noaccent).abs() > 1e-9]
    m["accent_differing_trials"] = int(len(trd))
    m["r2_accent_differing"] = float(smf.ols("d_target_logp ~ sim_raw", trd).fit().rsquared)
    m["r2_noaccent_differing"] = float(smf.ols("d_target_logp ~ sim_noaccent", trd).fit().rsquared)
    # H1: pseudoword dose-response.
    pw = df[df.word_type == "pseudoword"]
    mp = smf.ols("d_target_logp ~ sim_raw", pw).fit(cov_type="cluster", cov_kwds={"groups": pw["target_idx"]})
    m["pw_slope"], m["pw_slope_ci"] = float(mp.params["sim_raw"]), ci(mp.params["sim_raw"], mp.bse["sim_raw"])
    m["pw_slope_p"], m["pw_n"] = float(mp.pvalues["sim_raw"]), int(mp.nobs)
    # Anagrams against pseudowords whose spelling similarity lies within the central 80% of the anagrams'.
    an = df[df.word_type == "anagram"]
    lo, hi = an.sim_raw.quantile([0.1, 0.9])
    pwm = pw[(pw.sim_raw >= lo) & (pw.sim_raw <= hi)]
    m["anagram_vs_pseudoword"] = {"sim_range": [float(lo), float(hi)], "pseudoword_success": float(pwm.hit.mean()),
                                  "pseudoword_d": float(pwm.d_target_logp.mean()), "anagram_d": float(an.d_target_logp.mean())}
    # Change in the log-probability of the photographed (true) class, sign flipped so a drop is positive.
    m["true_class_drop"] = {wt: float(-df[df.word_type == wt].d_true_logp.mean()) for wt in ("english", "random")}
    # Share of rare low-similarity translation successes that come from compounds containing "web".
    low_hits = tr[(tr.sim_raw < 0.4) & (tr.hit == 1)]
    m["noncognate_hits"] = int(len(low_hits))
    m["noncognate_web_share"] = float(low_hits.word.str.contains("web").mean()) if len(low_hits) else None
    # Effects under each specification (the basic-specification SEs are recomputed below).
    for spec in ["basic", "position", "box"]:
        for eff in ["tr", "ff"]:
            est = float(s[f"{eff}_{spec}"])
            se = float(s[f"{eff}_{spec}_se"]) if f"{eff}_{spec}_se" in s.index else np.nan
            p = float(s[f"{eff}_{spec}_p"])
            m[f"{eff}_{spec}"] = {"est": est, "se": se, "ci": ci(est, se) if not np.isnan(se) else None, "p": p}
    m["ff_noother"] = {"est": float(s.ff_position_noother), "se": float(s.ff_position_noother_se),
                       "ci": ci(s.ff_position_noother, s.ff_position_noother_se), "p": float(s.ff_position_noother_p)}
    # That model is fitted to pseudowords and false friends alone (synthesis.ff_position_effect).
    noother, _ = load(key, exclude_other_class=True)
    noother = noother.dropna(subset=["d_target_logp"])
    m["ff_noother"]["n"] = int((noother.word_type.isin(["pseudoword", "false_friend"]) & (noother.sim_raw < 0.999)).sum())
    m["tr_vs_ff"] = {"est": float(s.tr_vs_ff), "se": float(s.tr_vs_ff_se), "ci": ci(s.tr_vs_ff, s.tr_vs_ff_se),
                     "p": float(s.tr_vs_ff_p), "n": int(s.tr_vs_ff_n)}
    facts["models"][key] = m

    # Figure 3 data (readers only): flip rate by spelling-similarity band and word type.
    if key in READERS:
        sub = df[df.word_type.isin(["pseudoword", "translation", "false_friend"]) & (df.sim_raw < 0.999)].copy()
        sub["band"] = pd.cut(sub.sim_raw, bands, labels=band_labels, include_lowest=True)
        g = sub.groupby(["word_type", "band"], observed=True).hit.agg(["mean", "size"]).reset_index()
        g["model"] = key
        fig3.append(g)
    fig4.append({"model": key, "tr": m["tr_box"]["est"], "tr_se": m["tr_box"]["se"],
                 "ff": m["ff_box"]["est"], "ff_se": m["ff_box"]["se"]})

# The basic-specification SEs were not stored in the summary file; recompute them here.
for key in ORDER:
    df, _ = load(key)
    df = df.dropna(subset=["d_target_logp"])
    sub = df[df.word_type.isin(["translation", "pseudoword", "false_friend"]) & (df.sim_raw < 0.999)].copy()
    sub["word_type"] = pd.Categorical(sub["word_type"], ["pseudoword", "translation", "false_friend"])
    mb = smf.ols("d_target_logp ~ bs(sim_raw, df=4) + C(word_type)", sub).fit(cov_type="cluster", cov_kwds={"groups": sub["target_idx"]})
    # The three specifications share this sample: observations and target-class clusters.
    facts["models"][key]["regression_n"] = int(mb.nobs)
    facts["models"][key]["regression_clusters"] = int(sub["target_idx"].nunique())
    for eff, t in [("tr", "translation"), ("ff", "false_friend")]:
        k = f"C(word_type)[T.{t}]"
        facts["models"][key][f"{eff}_basic"] = {"est": float(mb.params[k]), "se": float(mb.bse[k]),
                                                 "ci": ci(mb.params[k], mb.bse[k]), "p": float(mb.pvalues[k])}

# Size of the largest translation advantage the full specification cannot exclude, in three units.
for key in READERS:
    m = facts["models"][key]
    ul = m["tr_box"]["ci"][1]
    m["tr_box_upper"] = {"nats": float(ul), "fold": float(np.exp(ul)), "share_of_reading_index": float(ul / m["reading_index"]),
                         "letters_of_seven": float(7 * ul / m["pw_slope"])}

# Stimulus set composition (words that enter the analysis).
w = words_all[(words_all.review_status != "rejected") & ~words_all.ff_english_word.astype(bool)]
lang_name = {"en": "English", "es": "Spanish", "fr": "French", "pt": "Portuguese", "it": "Italian", "ca": "Catalan", "none": "None"}
comp = []
for wt, label in [("english", "English class name"), ("translation", "Translation"), ("false_friend", "False friend"),
                  ("pseudoword", "Pseudoword"), ("anagram", "Anagram"), ("random", "Random consonant string")]:
    g = w[w.word_type == wt]
    for lang in (["es", "fr", "pt", "it", "ca"] if wt in ("translation", "false_friend") else [g.lang.iloc[0]]):
        gl = g[g.lang == lang]
        if len(gl) == 0:
            continue
        comp.append({"type": label, "lang": lang_name[lang], "n": int(len(gl)), "sim": float(gl.sim_raw.mean())})
facts["stimuli"] = {
    "targets": int(w.target_idx.nunique()), "words_total": int(len(w)),
    "composition": comp,
    "run_total": int(sum(facts["models"][k]["stimuli_run"] for k in ORDER)),
    "analyzed_total": int(sum(facts["models"][k]["stimuli_analyzed"] for k in ORDER)),
}
# Translations that coincide with the class name or with each other, accents, and rendered images.
tr_w = w[w.word_type == "translation"].copy()
tr_w["en"] = tr_w.target_idx.map(w[w.word_type == "english"].set_index("target_idx").word)
nonid = tr_w[tr_w.word != tr_w.en]
shared = nonid[nonid.duplicated(["target_idx", "word"], keep=False)]
facts["stimuli"].update({
    "translations": int(len(tr_w)),
    "translations_identical": int((tr_w.word == tr_w.en).sum()),
    "translations_shared_strings": int(shared[["target_idx", "word"]].drop_duplicates().shape[0]),
    "translations_diacritic_differs": int(((tr_w.sim_raw - tr_w.sim_noaccent).abs() > 1e-9).sum()),
    "translations_accent_only": int(((tr_w.sim_noaccent > 0.999) & (tr_w.sim_raw < 0.999)).sum()),
    "words_rendered": int(len(words_all)),
    "images_rendered": int(len(man_all)),
    "images_distinct": int(man_all[["target_idx", "word", "source_idx"]].drop_duplicates().shape[0]),
    "name_length_mean": float(w[w.word_type == "english"].word.str.len().mean()),
    "pseudowords_per_target": {int(k): int(v) for k, v in
                               w[w.word_type == "pseudoword"].groupby("target_idx").size().value_counts().sort_index().items()},
    "targets_without_anagram": sorted(set(w[w.word_type == "english"].word) -
                                      set(w[w.word_type == "anagram"].target_en)),
})
# Exterior letters: share of translations and pseudowords with a similarity of 0.4 to below 1 that keep the first
# and the last letter of the English class name (lowercase, diacritics removed).
from common import strip_accents  # noqa: E402
ext = w[w.word_type.isin(["translation", "pseudoword"]) & (w.sim_raw >= 0.4) & (w.sim_raw < 0.999)].copy()
ext["plain"] = [strip_accents(x.lower()) for x in ext.word]
ext["en"] = ext.target_en.str.lower()
ext["first"] = [p[:1] == e[:1] for p, e in zip(ext.plain, ext.en)]
ext["last"] = [p[-1:] == e[-1:] for p, e in zip(ext.plain, ext.en)]
facts["stimuli"]["exterior_letters"] = {wt: {"first": float(g["first"].mean()), "last": float(g["last"].mean()),
                                             "n": int(len(g))} for wt, g in ext.groupby("word_type")}
# Review bookkeeping.
rv = words_all
facts["review"] = {
    "es_tr_reviewed": int(((rv.lang == "es") & (rv.word_type == "translation") & rv.review_status.isin(["accepted", "rejected"])).sum()),
    "es_tr_rejected": int(((rv.lang == "es") & (rv.word_type == "translation") & (rv.review_status == "rejected")).sum()),
    "es_ff_reviewed": int(((rv.lang == "es") & (rv.word_type == "false_friend") & rv.review_status.isin(["accepted", "rejected"])).sum()),
    "es_ff_rejected": int(((rv.lang == "es") & (rv.word_type == "false_friend") & (rv.review_status == "rejected")).sum()),
    "es_ff_replaced": int(((rv.lang == "es") & (rv.word_type == "false_friend") & (rv.review_status == "replacement")).sum()),
    "other_tr_reviewed": int((rv.lang.isin(["fr", "pt", "it", "ca"]) & (rv.word_type == "translation") & rv.review_status.isin(["accepted", "rejected"])).sum()),
    "other_tr_rejected": int((rv.lang.isin(["fr", "pt", "it", "ca"]) & (rv.word_type == "translation") & (rv.review_status == "rejected")).sum()),
    "other_tr_replaced": int((rv.lang.isin(["fr", "pt", "it", "ca"]) & (rv.word_type == "translation") & (rv.review_status == "replacement")).sum()),
    "fr_ff_reviewed": int(((rv.lang == "fr") & (rv.word_type == "false_friend") & rv.review_status.isin(["accepted", "rejected"])).sum()),
    "fr_ff_rejected": int(((rv.lang == "fr") & (rv.word_type == "false_friend") & (rv.review_status == "rejected")).sum()),
    "fr_ff_replaced": int(((rv.lang == "fr") & (rv.word_type == "false_friend") & (rv.review_status == "replacement")).sum()),
    "ff_english_excluded": int(((rv.word_type == "false_friend") & (rv.review_status != "rejected") & rv.ff_english_word.astype(bool)).sum()),
    "ff_other_class": int(((rv.word_type == "false_friend") & (rv.review_status != "rejected") & ~rv.ff_english_word.astype(bool) & rv.ff_other_class.astype(bool)).sum()),
}
# False friends kept from the review that fall outside the automatic selection thresholds.
ffs = w[w.word_type == "false_friend"].copy()
ffs["length_difference"] = (ffs.word.str.len() - ffs.target_en.str.len()).abs()
ffs["zipf"] = ffs.zipf_lang.astype(float)
outside = ffs[(ffs.sim_raw < 0.5) | (ffs.length_difference > 3) | (ffs.zipf < 2.5)]
facts["review"]["ff_outside_thresholds"] = [
    {"target": r.target_en, "word": r.word, "lang": r.lang, "sim": float(r.sim_raw), "length_difference": int(r.length_difference),
     "zipf": float(r.zipf), "source": r.source} for r in outside.itertuples()]
# Stability of the false-friend estimate (letter-position specification) across the lexical review.
pre = pd.read_csv(os.path.join(RES, "cross_model_summary_prereview.csv")).set_index("model")
facts["review"]["ff_position_before_after"] = {
    k: [float(pre.loc[k, "ff_position"]), float(summary.loc[k, "ff_position"])] for k in READERS if k in pre.index}

with open(os.path.join(PAPER, "data", "facts.json"), "w") as fh:
    json.dump(facts, fh, indent=1, ensure_ascii=False)
pd.concat(fig3).to_csv(os.path.join(PAPER, "data", "fig4_similarity.csv"), index=False)
pd.DataFrame(fig4).to_csv(os.path.join(PAPER, "data", "fig5_effects.csv"), index=False)
pd.DataFrame([{"model": k, "arch": facts["models"][k]["arch"], "corpus": facts["models"][k]["corpus"],
               "english_success": facts["models"][k]["english_success"], "reading_index": facts["models"][k]["reading_index"],
               "ri_lo": facts["models"][k]["ri_ci"][0], "ri_hi": facts["models"][k]["ri_ci"][1]} for k in ORDER]
             ).to_csv(os.path.join(PAPER, "data", "fig3_reading.csv"), index=False)

# ---------- LaTeX tables ----------
def nfmt(n):
    return f"{n:,}".replace(",", "{,}")


rows = []
for k in ORDER:
    m = facts["models"][k]
    rows.append(f"{m['arch']} & {m['corpus']} & {round(m['image_params_m'])} & {format(m['pairs_m'], ',').replace(',', '{,}')} & {m['photos_per_class']} & "
                f"{nfmt(m['stimuli_analyzed'])} & {100 * m['clean_top1']:.1f} \\\\")
table_models = r"""\begin{table}[tbp]
\centering
\caption{CLIP models evaluated. Parameters are those of the image encoder in millions, pairs are approximate training corpus sizes in millions of image-text pairs, stimuli are the images that entered the analysis, and clean top-1 is zero-shot accuracy on the 1{,}000 source photographs without text. The ViT-B/32 trained on LAION-5B pairs its image encoder with a multilingual text encoder.}
\label{tab:models}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}
\toprule
Encoder & Training data & Parameters & Pairs & Photos per & Stimuli & Clean top-1 \\
 & & (M) & (M) & target & & (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
with open(os.path.join(PAPER, "tables", "table_models.tex"), "w") as fh:
    fh.write(table_models)

rows, last = [], None
for c in comp:
    t = c["type"] if c["type"] != last else ""
    last = c["type"]
    rows.append(f"{t} & {c['lang']} & {nfmt(c['n'])} & {c['sim']:.2f} \\\\")
table_stimuli = r"""\begin{table}[tbp]
\centering
\caption{Written words per word type and language after lexical review. Similarity is the mean normalized Levenshtein similarity to the English class name.}
\label{tab:stimuli}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrr@{}}
\toprule
Word type & Language & Words & Mean similarity \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
Total & & """ + nfmt(len(w)) + r""" & \\
\bottomrule
\end{tabular*}
\end{table}
"""
with open(os.path.join(PAPER, "tables", "table_stimuli.tex"), "w") as fh:
    fh.write(table_stimuli)


print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else {x: (round(y, 4) if isinstance(y, float) else y) for x, y in vv.items()})
                      for kk, vv in v.items() if kk in ("stimuli_analyzed", "english_success", "noncognate_success", "cognate_success",
                                                       "identical_success", "pw_slope", "tr_box", "ff_box", "r2_accent", "r2_noaccent")}
                  for k, v in facts["models"].items()}, indent=1, default=str)[:6000])
print(json.dumps(facts["stimuli"], indent=1)[:3000])
print(json.dumps(facts["review"], indent=1))
