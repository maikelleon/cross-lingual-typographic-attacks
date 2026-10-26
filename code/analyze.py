"""First-pass analysis for one or more models.

Usage: python3 analyze.py vitb32_laion400m [more model keys]
Prints: clean accuracy; attack success and target log-probability change by word type and
language; H1 (pseudoword success vs spelling similarity); H2 (translations vs pseudowords and
false friends at matched spelling similarity), with standard errors clustered by target class.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder


def load(key, exclude_other_class=False):
    """Stimulus-level results for one model, restricted to words that passed review.

    Excluded: words the reviewer rejected (their replacements are included instead) and false
    friends that are also English words. exclude_other_class=True also drops false friends whose
    meaning names another ImageNet class (sensitivity check).
    """
    words = pd.read_csv(os.path.join(ROOT, "stimuli", "words.csv"))
    keep = (words["review_status"] != "rejected") & ~words["ff_english_word"].astype(bool)
    if exclude_other_class:
        keep &= ~words["ff_other_class"].astype(bool)
    words = words[keep]
    man = pd.read_csv(os.path.join(ROOT, "stimuli", "manifest.csv"))
    res = pd.read_csv(os.path.join(ROOT, "results", f"{key}_stimuli.csv")).drop_duplicates("stim_id")
    df = man.merge(res, on="stim_id").merge(
        words[["target_idx", "word_type", "lang", "word", "sim_raw", "sim_noaccent", "n_chars", "zipf_en", "zipf_lang",
               "substitutions", "source", "review_status", "ff_other_class"]],
        on=["target_idx", "word_type", "lang", "word"], how="inner")
    clean = np.load(os.path.join(ROOT, "results", f"{key}_clean.npz"))["logp"]
    df["hit"] = (df["pred"] == df["target_idx"]).astype(int)
    df["source_clean_correct"] = (clean.argmax(1)[df["source_idx"]] == df["source_idx"]).astype(int)
    # Any overlaid text flattens the class distribution and lifts every class a little, so effects are
    # measured against the random consonant string written on the same photo for the same target.
    base = (df[df.word_type == "random"].groupby(["target_idx", "source_idx"])["d_target_logp"]
            .mean().rename("d_random"))
    df = df.merge(base, on=["target_idx", "source_idx"], how="left")
    df["d_target_logp_raw"] = df["d_target_logp"]
    df["d_target_logp"] = df["d_target_logp_raw"] - df["d_random"]
    return df, clean


class NoFit:
    """Stand-in when a logistic model cannot be estimated (too few successes, e.g. a model that does not read)."""
    params = bse = pvalues = rsquared = None


def fit(formula, data, family="logit"):
    # Rows without a random-string baseline (a target still mid-run) are dropped so groups align.
    data = data.dropna(subset=["d_target_logp"])
    groups = data["target_idx"]
    if family == "logit" and data["hit"].sum() < 30:
        return NoFit()
    if family == "logit":
        m = smf.logit(formula, data).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    else:
        m = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return m


def report(key):
    df, clean = load(key)
    print(f"\n===== {key} =====")
    print(f"stimuli analyzed: {len(df):,}   clean top-1 on 1,000 photos: {(clean.argmax(1) == np.arange(1000)).mean():.3f}")
    t = (df.groupby(["word_type"]).agg(n=("hit", "size"), success=("hit", "mean"),
                                       d_logp=("d_target_logp", "mean"), median_rank=("target_rank", "median"))
         .sort_values("success", ascending=False))
    print("\nBy word type (success = prediction becomes the written word's target class):")
    print(t.round(3).to_string())
    tl = (df[df.word_type.isin(["translation", "false_friend"])]
          .groupby(["word_type", "lang"]).agg(n=("hit", "size"), success=("hit", "mean"),
                                              sim=("sim_raw", "mean"), d_logp=("d_target_logp", "mean")))
    print("\nTranslations and false friends by language:")
    print(tl.round(3).to_string())

    # Reading index: how much more an English label lifts its class than a random string on the same
    # photo (nats), with a 95% CI from a bootstrap over target classes. Near zero = the model does not read.
    en = df[df.word_type == "english"].groupby("target_idx")["d_target_logp"].mean()
    rng = np.random.default_rng(0)
    boots = [en.sample(len(en), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 2000)]
    print(f"\nReading index (English vs random string): {en.mean():.2f} nats, 95% CI [{np.percentile(boots, 2.5):.2f}, {np.percentile(boots, 97.5):.2f}]")

    # H1: among meaningless pseudowords, success rises with spelling similarity to the English name.
    pw = df[df.word_type == "pseudoword"]
    bins = pd.cut(pw["sim_raw"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    print("\nH1 pseudowords: success by spelling similarity to the English name")
    print(pw.groupby(bins, observed=True).agg(n=("hit", "size"), success=("hit", "mean"), d_logp=("d_target_logp", "mean")).round(3).to_string())
    m1 = fit("hit ~ sim_raw", pw)
    m1b = fit("d_target_logp ~ sim_raw", pw, family="ols")
    logit_txt = ("logit not estimable (too few successes)" if m1.params is None else
                 f"logit slope on similarity: {m1.params['sim_raw']:.2f} (SE {m1.bse['sim_raw']:.2f}, p={m1.pvalues['sim_raw']:.2g})")
    print(f"{logit_txt};  OLS slope on log-prob change: {m1b.params['sim_raw']:.2f} (SE {m1b.bse['sim_raw']:.2f}, p={m1b.pvalues['sim_raw']:.2g})")

    # H2: at matched spelling similarity, do real translations beat meaningless strings and false friends?
    sub = df[df.word_type.isin(["translation", "pseudoword", "false_friend"])].copy()
    sub["word_type"] = pd.Categorical(sub["word_type"], ["pseudoword", "translation", "false_friend"])
    m2 = fit("d_target_logp ~ sim_raw + C(word_type)", sub, family="ols")
    print("\nH2 (OLS on target log-prob change, controlling for spelling similarity; reference = pseudoword):")
    for name in m2.params.index:
        print(f"  {name:<36} {m2.params[name]:7.3f}  SE {m2.bse[name]:.3f}  p={m2.pvalues[name]:.2g}")
    m2l = fit("hit ~ sim_raw + C(word_type)", sub)
    if m2l.params is None:
        print("H2 (logit on success): not estimable (too few successes)")
    else:
        print("H2 (logit on success):")
        for name in m2l.params.index:
            print(f"  {name:<36} {m2l.params[name]:7.3f}  SE {m2l.bse[name]:.3f}  p={m2l.pvalues[name]:.2g}")

    # Accents: does similarity that ignores accents explain translation effects better?
    tr = df[df.word_type == "translation"]
    a = fit("d_target_logp ~ sim_raw", tr, family="ols")
    b = fit("d_target_logp ~ sim_noaccent", tr, family="ols")
    print(f"\nTranslations: R^2 with accent-sensitive similarity {a.rsquared:.3f} vs accent-blind {b.rsquared:.3f}")
    return df


if __name__ == "__main__":
    for key in sys.argv[1:]:
        report(key)
