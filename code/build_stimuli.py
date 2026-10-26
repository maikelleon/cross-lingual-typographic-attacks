"""Build the stimulus set for the cross-lingual typographic study.

For each eligible ImageNet target class t (English name E):
  english      E itself (reference attack)
  translation  the most frequent lemma of t's synset in Spanish, French, Portuguese, Italian, Catalan
               (Open Multilingual WordNet; frequency from wordfreq)
  false_friend a common real word in Spanish or French that is spelled like E but is unrelated in
               meaning (WordNet path similarity < 0.1); flagged for human validation
  pseudoword   E with a graded share of letters substituted (vowel->vowel, consonant->consonant),
               checked against real words, giving controlled spelling similarity without meaning
  anagram      E's letters reordered (letter identity kept, order destroyed)
  random       a consonant string of E's length (floor condition)
Every word for t is written on the same M source photos (paired design).

Outputs: stimuli/words.csv (one row per target x word), stimuli/manifest.csv (one row per
stimulus image), stimuli/contact_sheet.png (random sample for visual checking).
"""
import csv
import json
import os
import random
import re
from collections import Counter

import nltk
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from wordfreq import zipf_frequency

from common import ROOT, image_files, load_source, ortho_sim, overlay
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES

nltk.data.path.insert(0, os.path.join(ROOT, "data", "nltk_data"))
from nltk.corpus import wordnet as wn  # noqa: E402

SEED = 20261024
M_IMAGES = 16                      # source photos per target
LANGS = {"spa": "es", "fra": "fr", "por": "pt", "ita": "it", "cat": "ca"}
FF_LANGS = ["spa", "fra"]          # languages for mined false friends
PSEUDO_FRACTIONS = [0.15, 0.3, 0.5, 0.7, 0.9]
VOWELS, CONSONANTS = "aeiou", "bcdfghjklmnpqrstvwxz"
WORD_RE = re.compile(r"^[^\W\d_]+(?:[ '-][^\W\d_]+){0,2}$")   # letters, up to three words
OUT = os.path.join(ROOT, "stimuli")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(SEED)

files = image_files()
wnids = [os.path.basename(f).split("_")[0] for f in files]
names = [n.lower() for n in IMAGENET_CLASSNAMES]
name_counts = Counter(names)
synsets = [wn.synset_from_pos_and_offset("n", int(w[1:])) for w in wnids]


def clean(lemma):
    return lemma.replace("_", " ").strip().lower()


# Real-word lexicon used to reject accidental real words among pseudowords and anagrams.
english_words = {clean(l) for l in wn.all_lemma_names()}
lexicon = english_words | set(names)
lang_lemmas, proper_nouns = {}, {}
for code in LANGS:
    raw = list(wn.all_lemma_names(lang=code))
    lemmas = {clean(l) for l in raw}
    lower_forms = {clean(l) for l in raw if not l[:1].isupper()}
    lang_lemmas[code] = lemmas
    proper_nouns[code] = lemmas - lower_forms          # only ever capitalized: names of people and places
    lexicon |= lemmas


def is_ok_word(w):
    return bool(WORD_RE.match(w)) and len(w) <= 20


def ancestor_lemmas(syn, code, depth=4):
    """Lemmas (in `code`) of the synset's hypernyms, so generic words like 'ave' (bird) are not used."""
    out, frontier = set(), [syn]
    for _ in range(depth):
        frontier = [h for s in frontier for h in s.hypernyms() + s.instance_hypernyms()]
        for h in frontier:
            out |= {clean(l) for l in h.lemma_names(code)}
    return out


def translations(t):
    """Per language: a lemma of t's synset that is not a proper noun or a hypernym's name;
    single words preferred, then the most frequent in that language."""
    out = {}
    for code, wf in LANGS.items():
        generic = ancestor_lemmas(synsets[t], code)
        cands = [clean(l) for l in synsets[t].lemma_names(code)]
        cands = [c for c in dict.fromkeys(cands)
                 if is_ok_word(c) and c not in proper_nouns[code] and c not in generic]
        if cands:
            out[code] = max(cands, key=lambda c: (" " not in c, zipf_frequency(c, wf), -cands.index(c)))
    return out


def substitute(word, k):
    for _ in range(200):
        pos = rng.sample(range(len(word)), k)
        chars = list(word)
        for p in pos:
            pool = VOWELS if chars[p] in VOWELS else CONSONANTS
            chars[p] = rng.choice([c for c in pool if c != chars[p]])
        cand = "".join(chars)
        if cand not in lexicon:
            return cand
    return None


def anagram(word):
    if len(set(word)) < 2:
        return None
    for _ in range(200):
        cand = "".join(rng.sample(word, len(word)))
        if cand != word and cand not in lexicon:
            return cand
    return None


def random_string(n):
    for _ in range(200):
        cand = "".join(rng.choice(CONSONANTS) for _ in range(n))
        if cand not in lexicon:
            return cand
    return None


def false_friend(t, english, code, banned):
    """Most English-like common word in `code` that is unrelated in meaning to class t."""
    wf = LANGS[code]
    # Sorted so that ties are broken the same way in every run (Python randomizes set order per process).
    pool = sorted(w for w in lang_lemmas[code] if " " not in w and w.isalpha() and abs(len(w) - len(english)) <= 3)
    hits = process.extract(english, pool, scorer=Levenshtein.normalized_similarity,
                           score_cutoff=0.5, limit=40)
    best = None
    for cand, score, _ in hits:
        if cand == english or cand in banned or cand in names or cand in proper_nouns[code]:
            continue
        # Must be a common word in its own language and not an English word, so any effect
        # cannot come from CLIP reading it as English.
        if zipf_frequency(cand, wf) < 2.5 or cand in english_words or zipf_frequency(cand, "en") >= 3.0:
            continue
        related = [synsets[t].path_similarity(s) or 0 for s in wn.synsets(cand, lang=code, pos="n")]
        if related and max(related) >= 0.1:
            continue
        key = (score, zipf_frequency(cand, wf))
        if best is None or key > best[0]:
            best = (key, cand)
    return best[1] if best else None


# Eligible targets: a single-word English name, unique among the 1,000 classes,
# with translations in at least two of the five languages.
targets = []
for t, en in enumerate(names):
    if not (en.isalpha() and en.isascii() and 3 <= len(en) <= 12 and name_counts[en] == 1):
        continue
    tr = translations(t)
    if len(tr) >= 2:
        targets.append((t, en, tr))
print(f"eligible targets: {len(targets)}")

word_rows = []
for t, en, tr in targets:
    rows = [("english", "en", en, None)]
    for code, w in tr.items():
        rows.append(("translation", LANGS[code], w, None))
    banned = set(tr.values())
    for code in FF_LANGS:
        ff = false_friend(t, en, code, banned)
        if ff:
            rows.append(("false_friend", LANGS[code], ff, None))
    seen_k = set()
    for frac in PSEUDO_FRACTIONS:
        k = max(1, round(frac * len(en)))
        if k in seen_k or k > len(en):
            continue
        seen_k.add(k)
        pw = substitute(en, k)
        if pw:
            rows.append(("pseudoword", "none", pw, k))
    ag = anagram(en)
    if ag:
        rows.append(("anagram", "none", ag, None))
    rs = random_string(len(en))
    if rs:
        rows.append(("random", "none", rs, None))
    for wtype, lang, w, k in rows:
        wf = lang if lang in ("es", "fr", "pt", "it", "ca") else None
        word_rows.append({
            "target_idx": t, "target_wnid": wnids[t], "target_en": en,
            "word_type": wtype, "lang": lang, "word": w, "substitutions": k if k is not None else "",
            "sim_raw": round(ortho_sim(w, en, accents=True), 4),
            "sim_noaccent": round(ortho_sim(w, en, accents=False), 4),
            "n_chars": len(w),
            "zipf_en": zipf_frequency(w, "en"),
            "zipf_lang": zipf_frequency(w, wf) if wf else "",
        })

# Paired design: every word for target t is written on the same M photos (never class t itself).
manifest = []
sources = {}
for t, en, tr in targets:
    pool = [i for i in range(1000) if i != t]
    sources[t] = rng.sample(pool, M_IMAGES)
sid = 0
for wr in word_rows:
    # source_rank lets slower models use the first 10 photos per target while ViT-B/32 uses all 16.
    for rank, s in enumerate(sources[wr["target_idx"]]):
        manifest.append({"stim_id": sid, **{k: wr[k] for k in ("target_idx", "target_en", "word_type", "lang", "word")},
                         "source_idx": s, "source_wnid": wnids[s], "source_rank": rank})
        sid += 1

with open(os.path.join(OUT, "words_prereview.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(word_rows[0]))
    w.writeheader()
    w.writerows(word_rows)
with open(os.path.join(OUT, "manifest_prereview.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
    w.writeheader()
    w.writerows(manifest)

summary = {
    "seed": SEED, "targets": len(targets), "photos_per_target": M_IMAGES,
    "words": len(word_rows), "stimuli": len(manifest),
    "words_by_type": dict(Counter(r["word_type"] for r in word_rows)),
    "translations_by_lang": dict(Counter(r["lang"] for r in word_rows if r["word_type"] == "translation")),
    "false_friends_by_lang": dict(Counter(r["lang"] for r in word_rows if r["word_type"] == "false_friend")),
}
with open(os.path.join(ROOT, "outputs", "build_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))

# Contact sheet: random stimuli across word types, for visual checking of rendering.
from PIL import Image, ImageDraw  # noqa: E402
pick = rng.sample(manifest, 24)
sheet = Image.new("RGB", (6 * 224, 4 * 250), "white")
d = ImageDraw.Draw(sheet)
for i, m in enumerate(pick):
    img = overlay(load_source(files[m["source_idx"]]), m["word"])
    x, y = (i % 6) * 224, (i // 6) * 250
    sheet.paste(img, (x, y))
    d.text((x + 4, y + 228), f'{m["word_type"]}/{m["lang"]}: {m["target_en"]}', fill="black")
sheet.save(os.path.join(ROOT, "outputs", "figures", "contact_sheet.png"))
