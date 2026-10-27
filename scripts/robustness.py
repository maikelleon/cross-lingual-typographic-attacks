"""Robustness checks for the main effects in the four reading models.

1. Spelling gradient within translations of each language: slope of the change in the target
   log-probability on spelling similarity (OLS, standard errors clustered by target class).
2. Translation and false-friend effects against pseudowords under the full specification (spelling
   spline, letter-position covariates, text-box size), estimated separately for each language.
3. The same effects in two random halves of the target classes (split-half replication).
4. The pooled full-specification effects without the translations that differ from the English class name only
   in diacritics, and with each translation string counted once per target and photograph.
5. The translation effect among words with a spelling similarity below 0.4, a direct test of meaning without
   shared spelling, and two paired comparisons estimated jointly on the stimuli both models classified: the
   multilingual ViT-B/32 against the ViT-B/32 trained on LAION-400M (caption language), and the ViT-L/14
   against the RN50, both trained on OpenAI WIT (scale). Each reports the differences in the translation and
   false-friend effects, the difference between those two differences, and the difference in the
   low-similarity effect.

Outputs: data/robustness.json and tables/table_effects.tex, which combines the pooled estimates in
data/facts.json with checks 2 and 3 (run compute_facts.py first). The gradients of check 1 are in a different
unit, so they are reported in the text and in data/robustness.json rather than in the table.
Usage: python3 scripts/robustness.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE) if os.path.isdir(os.path.join(os.path.dirname(HERE), "code")) \
    else os.path.dirname(os.path.dirname(HERE))
CODE = os.path.join(PROJECT, "code")
sys.path.insert(0, CODE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from rapidfuzz.distance import JaroWinkler  # noqa: E402

from analyze import load  # noqa: E402
from common import strip_accents  # noqa: E402
from synthesis import box_adjusted, dice, prefix_ratio, suffix_ratio, text_box  # noqa: E402

OUT = os.path.join(PROJECT, "outputs") if os.path.isdir(os.path.join(PROJECT, "outputs")) and \
    not os.path.isdir(os.path.join(PROJECT, "paper")) else os.path.join(PROJECT, "paper")
READERS = ["vitb32_laion400m", "vitb16_laion400m", "rn50_openai", "vitl14_openai", "vitb32_xlmr_laion5b"]
MULTILINGUAL = "vitb32_xlmr_laion5b"
LANGS = [("es", "Spanish"), ("fr", "French"), ("pt", "Portuguese"), ("it", "Italian"), ("ca", "Catalan")]
FORMULA = ("d_target_logp ~ bs(sim_raw, df=4) + bs(jw, df=3) + prefix + suffix + dice + bs(dw, df=3) + box_h"
           " + C(word_type)")


def covariates(df):
    """Spelling, letter-position, and text-box covariates for the non-identical words of one model."""
    sub = df[df.word_type.isin(["pseudoword", "translation", "false_friend"]) & (df.sim_raw < 0.999)].copy()
    rnd = df[df.word_type == "random"].drop_duplicates(["target_idx"])[["target_idx", "word"]]
    sub = sub.merge(rnd.rename(columns={"word": "rnd"}), on="target_idx", how="left")
    sub["box_h"] = [text_box(w)[1] for w in sub.word]
    sub["dw"] = [text_box(w)[0] - text_box(r)[0] for w, r in zip(sub.word, sub.rnd)]
    sub["prefix"] = [prefix_ratio(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["suffix"] = [suffix_ratio(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["dice"] = [dice(w, e) for w, e in zip(sub.word, sub.target_en)]
    sub["jw"] = [JaroWinkler.normalized_similarity(strip_accents(w.lower()), e) for w, e in zip(sub.word, sub.target_en)]
    return sub


def effect(sub, word_type):
    """Effect of one word type against pseudowords under the full specification."""
    d = sub[sub.word_type.isin(["pseudoword", word_type])].copy()
    d["word_type"] = pd.Categorical(d["word_type"], ["pseudoword", word_type])
    m = smf.ols(FORMULA, d).fit(cov_type="cluster", cov_kwds={"groups": d["target_idx"]})
    k = f"C(word_type)[T.{word_type}]"
    est, se = float(m.params[k]), float(m.bse[k])
    return {"est": est, "se": se, "ci": [est - 1.96 * se, est + 1.96 * se], "p": float(m.pvalues[k]),
            "n_words": int(d[d.word_type == word_type][["target_idx", "word"]].drop_duplicates().shape[0]),
            "n": int(m.nobs)}


def ff_language_difference(sub):
    """French minus Spanish false-friend effect, both against pseudowords in one full-specification model,
    with a two-sided Wald test on the clustered covariance."""
    d = sub[sub.word_type.isin(["pseudoword", "false_friend"])].copy()
    d["word_type"] = pd.Categorical(np.where(d.word_type == "pseudoword", "pseudoword", "ff_" + d.lang),
                                    ["pseudoword", "ff_es", "ff_fr"])
    m = smf.ols(FORMULA, d).fit(cov_type="cluster", cov_kwds={"groups": d["target_idx"]})
    names = list(m.params.index)
    contrast = np.zeros(len(names))
    contrast[names.index("C(word_type)[T.ff_fr]")], contrast[names.index("C(word_type)[T.ff_es]")] = 1.0, -1.0
    t = m.t_test(contrast)
    est, se = float(np.squeeze(t.effect)), float(np.squeeze(t.sd))
    return {"est": est, "se": se, "ci": [est - 1.96 * se, est + 1.96 * se], "p": float(np.squeeze(t.pvalue)),
            "n": int(m.nobs)}


def low_similarity(sub, limit=0.4):
    """Translation effect against pseudowords among words that share little spelling with the class name
    (similarity below the limit), under the full specification."""
    return effect(sub[sub.word_type.isin(["pseudoword", "translation"]) & (sub.sim_raw < limit)], "translation")


def pair_difference(sub_first, sub_second, limit=0.4):
    """Differences between two models (second minus first) from full-specification models in which every term
    interacts with the model. Both models classified the same stimuli, so standard errors clustered by target
    class allow for the pairing. Reports the differences in the translation and false-friend effects, the
    difference between them (the part of the change specific to translations), and the difference in the
    translation effect among words below the similarity limit."""
    rhs = FORMULA.split("~", 1)[1]

    def fit(first, second):
        both = pd.concat([first.assign(model="first"), second.assign(model="second")], ignore_index=True)
        both["model"] = pd.Categorical(both["model"], ["first", "second"])
        types = [t for t in ["pseudoword", "translation", "false_friend"] if t in set(both.word_type)]
        both["word_type"] = pd.Categorical(both["word_type"], types)
        return smf.ols(f"d_target_logp ~ ({rhs}) * C(model)", both).fit(cov_type="cluster",
                                                                       cov_kwds={"groups": both["target_idx"]})

    def test(m, plus, minus=None):
        names = list(m.params.index)
        contrast = np.zeros(len(names))
        contrast[names.index(plus)] = 1.0
        if minus is not None:
            contrast[names.index(minus)] = -1.0
        t = m.t_test(contrast)
        est, se = float(np.squeeze(t.effect)), float(np.squeeze(t.sd))
        return {"est": est, "se": se, "ci": [est - 1.96 * se, est + 1.96 * se], "p": float(np.squeeze(t.pvalue)),
                "n": int(m.nobs)}

    key = "C(word_type)[T.{}]:C(model)[T.second]".format
    m = fit(sub_first, sub_second)
    out = {"tr": test(m, key("translation")), "ff": test(m, key("false_friend")),
           "tr_minus_ff": test(m, key("translation"), key("false_friend"))}

    def low(s):
        return s[s.word_type.isin(["pseudoword", "translation"]) & (s.sim_raw < limit)]

    out["tr_low_similarity"] = test(fit(low(sub_first), low(sub_second)), key("translation"))
    return out


def slope(d):
    m = smf.ols("d_target_logp ~ sim_raw", d).fit(cov_type="cluster", cov_kwds={"groups": d["target_idx"]})
    est, se = float(m.params["sim_raw"]), float(m.bse["sim_raw"])
    return {"est": est, "se": se, "ci": [est - 1.96 * se, est + 1.96 * se], "p": float(m.pvalues["sim_raw"]),
            "n": int(m.nobs)}


def main():
    res, subs = {}, {}
    for key in READERS:
        df, _ = load(key)
        df = df.dropna(subset=["d_target_logp"])
        sub = covariates(df)
        targets = np.array(sorted(df.target_idx.unique()))
        half_a = set(np.random.default_rng(20261026).permutation(targets)[: len(targets) // 2].tolist())
        r = {"slope": {}, "tr": {}, "ff": {}}
        tr_all = df[df.word_type == "translation"]
        for code, _ in LANGS:
            r["slope"][code] = slope(tr_all[tr_all.lang == code])
            lang_sub = sub[(sub.word_type == "pseudoword") | ((sub.word_type == "translation") & (sub.lang == code))]
            r["tr"][code] = effect(lang_sub, "translation")
            if code in ("es", "fr"):
                lang_sub = sub[(sub.word_type == "pseudoword") | ((sub.word_type == "false_friend") & (sub.lang == code))]
                r["ff"][code] = effect(lang_sub, "false_friend")
        r["ff_lang_diff"] = ff_language_difference(sub)
        r["tr_low_similarity"] = low_similarity(sub)
        subs[key] = sub
        for name, keep in [("half_a", sub.target_idx.isin(half_a)), ("half_b", ~sub.target_idx.isin(half_a))]:
            r["tr"][name] = effect(sub[keep], "translation")
            r["ff"][name] = effect(sub[keep], "false_friend")
        accent_only = (df.word_type == "translation") & (df.sim_noaccent > 0.999) & (df.sim_raw < 0.999)
        repeated = (df.word_type == "translation") & df.duplicated(["target_idx", "word", "source_idx"], keep="first")
        r["sensitivity"] = {}
        for name, keep in [("without_accent_only", ~accent_only), ("strings_once", ~repeated)]:
            b = box_adjusted(df[keep])
            r["sensitivity"][name] = {
                eff: {"est": float(b[f"{eff}_box"]), "se": float(b[f"{eff}_box_se"]),
                      "ci": [float(b[f"{eff}_box"] - 1.96 * b[f"{eff}_box_se"]), float(b[f"{eff}_box"] + 1.96 * b[f"{eff}_box_se"])],
                      "p": float(b[f"{eff}_box_p"])} for eff in ("tr", "ff")}
        res[key] = r
        print(key, "done", flush=True)
    # The two controlled comparisons among reading models: caption language (two ViT-B/32 models with the same
    # architecture and about the same number of training samples) and scale (two models trained on OpenAI WIT).
    res["twin_difference"] = pair_difference(subs["vitb32_laion400m"], subs[MULTILINGUAL])
    res["scale_difference"] = pair_difference(subs["rn50_openai"], subs["vitl14_openai"])
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    with open(os.path.join(OUT, "data", "robustness.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_table(res)


def fmt(x):
    s = f"{x:.3f}" if 0 < abs(x) < 0.005 else f"{x:.2f}"   # keep a bound that rounds to zero distinguishable from it
    return s.replace("-", "\\textminus ") if s.startswith("-") else s


def cell(d):
    return f"{fmt(d['est'])} [{fmt(d['ci'][0])}, {fmt(d['ci'][1])}]"


def write_table(res):
    pooled = json.load(open(os.path.join(OUT, "data", "facts.json")))["models"]
    rows = []
    # With five or more models an estimate and its interval no longer fit side by side within the text width,
    # so each interval moves to its own line under the estimate.
    stack = len(READERS) >= 5

    def head(text):
        rows.append(f"\\multicolumn{{{len(READERS) + 1}}}{{c}}{{{text}}} \\\\")

    def row(label, ds, indent="\\quad"):
        if stack:
            rows.append(f"{indent} {label} & " + " & ".join(fmt(d["est"]) for d in ds) + " \\\\")
            rows.append(" & " + " & ".join(f"[{fmt(d['ci'][0])}, {fmt(d['ci'][1])}]" for d in ds) + " \\\\[2pt]")
        else:
            rows.append(f"{indent} {label} & " + " & ".join(cell(d) for d in ds) + " \\\\")

    def subsets(kind, codes):
        names = dict(LANGS + [("half_a", "Target half A"), ("half_b", "Target half B")])
        for code in codes:
            row(names[code], [res[k][kind][code] for k in READERS], "\\qquad")

    head("Translation vs.\\ pseudoword")
    for label, key in [("Similarity spline", "tr_basic"), ("Letter position", "tr_position"),
                       ("Full specification", "tr_box")]:
        row(label, [pooled[k][key] for k in READERS])
    subsets("tr", [c for c, _ in LANGS] + ["half_a", "half_b"])
    row("Similarity below 0.4", [res[k]["tr_low_similarity"] for k in READERS], "\\qquad")
    head("False friend vs.\\ pseudoword")
    for label, key in [("Similarity spline", "ff_basic"), ("Letter position", "ff_position"),
                       ("No other-class words", "ff_noother"), ("Full specification", "ff_box")]:
        row(label, [pooled[k][key] for k in READERS])
    subsets("ff", ["es", "fr", "half_a", "half_b"])
    head("Translation vs.\\ false friend")
    row("Direct comparison", [pooled[k]["tr_vs_ff"] for k in READERS])
    # The spelling gradients within each language (res[k]["slope"]) are in nats per unit similarity, a different
    # unit from every row above, so they are reported in the text and in data/robustness.json, not in this table.
    def num(n):
        return f"{n:,}".replace(",", "{,}")

    def by_photos(get):
        """One count per number of photographs per target (16 or 10), largest first."""
        counts = {}
        for k in READERS:
            counts.setdefault(pooled[k]["photos_per_class"], set()).add(get(k))
        assert all(len(v) == 1 for v in counts.values())
        return [num(counts[p].pop()) for p in sorted(counts, reverse=True)]

    words = {3: "three", 4: "four", 5: "five"}
    n_pooled = by_photos(lambda k: pooled[k]["regression_n"])
    n_direct = by_photos(lambda k: pooled[k]["tr_vs_ff"]["n"])
    n_noother = by_photos(lambda k: pooled[k]["ff_noother"]["n"])
    n_low = by_photos(lambda k: res[k]["tr_low_similarity"]["n"])
    clusters = num(pooled[READERS[0]]["regression_clusters"])
    lang_n = [res[k][kind][c]["n"] for k in READERS for kind, codes in [("tr", [c for c, _ in LANGS]), ("ff", ["es", "fr"])]
              for c in codes]
    half_n = [res[k][kind][h]["n"] for k in READERS for kind in ("tr", "ff") for h in ("half_a", "half_b")]
    cols = len(READERS)
    # Column groups by training corpus: the English-centered readers first, then the multilingual one.
    groups = [("English-centered", [k for k in READERS if k != MULTILINGUAL]), ("Multilingual", [MULTILINGUAL])]
    order = [k for _, keys in groups for k in keys]
    assert order == READERS
    head_groups, rules, start = [], [], 2
    for name, keys in groups:
        head_groups.append(f"\\multicolumn{{{len(keys)}}}{{c}}{{{name}}}")
        rules.append(f"\\cmidrule({'lr' if start + len(keys) - 1 < cols + 1 else 'l'}){{{start}-{start + len(keys) - 1}}}")
        start += len(keys)
    head_arch = " & ".join(pooled[k]["arch"] for k in READERS)
    head_corpus = " & ".join(pooled[k]["corpus"].replace(" (multilingual)", "") for k in READERS)
    table = r"""\begin{table}[tbp]
\centering
\caption{Effects of meaning on the log-probability of the target class in the """ + words[cols] + r""" reading models, grouped by training corpus as in Section~\ref{sec:models}. Values are differences between word types with 95\% CIs""" + (" (in brackets, under each value)" if stack else "") + r""" from standard errors clustered by target class; an interval that excludes zero corresponds to $P < 0.05$ in a two-sided Wald test. The pooled comparisons with pseudowords use """ + n_pooled[0] + r""" observations for the models evaluated on 16 photographs per target and """ + n_pooled[1] + r""" for those evaluated on 10, with """ + clusters + r""" target classes, the row without other-class words """ + n_noother[0] + " and " + n_noother[1] + r""", the language-specific rows """ + num(min(lang_n)) + " to " + num(max(lang_n)) + r""", the half-sample rows """ + num(min(half_n)) + " to " + num(max(half_n)) + r""", the low-similarity row """ + n_low[0] + " and " + n_low[1] + r""", and the direct comparison """ + n_direct[0] + " and " + n_direct[1] + r""". Specifications follow Section~\ref{sec:analysis}. The row without other-class words repeats the letter-position specification on pseudowords and false friends alone, after removing the false friends that name another ImageNet class, and the rows indented under the full specification re-estimate it for one language, for one random half of the target classes, or for the words whose spelling similarity to the class name is below 0.4.}
\label{tab:effects}
\footnotesize
\setlength{\tabcolsep}{3pt}
""" + ("\\renewcommand{\\arraystretch}{0.92}\n" if stack else "") + r"""\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l""" + "c" * cols + r"""@{}}
\toprule
 & \multicolumn{""" + str(cols) + r"""}{c}{Difference in log-probability (nats)} \\
\cmidrule(l){2-""" + str(cols + 1) + r"""}
 & """ + " & ".join(head_groups) + r""" \\
""" + "".join(rules) + r"""
Contrast and & """ + head_arch + r""" \\
specification & """ + head_corpus + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    with open(os.path.join(OUT, "tables", "table_effects.tex"), "w") as fh:
        fh.write(table)


if __name__ == "__main__":
    if "--table-only" in sys.argv:
        write_table(json.load(open(os.path.join(OUT, "data", "robustness.json"))))
    else:
        main()
