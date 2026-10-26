"""Apply the reviewer's decisions on the Spanish translations and false friends.

Rejected rows stay in words.csv (and their results stay in the result files) but are marked
review_status='rejected' and excluded from analysis. Each reviewer replacement becomes a new word
row (source='reviewer') with new stimulus IDs on the same source photos as the rest of its target,
so the next pass of run_models.py runs only the new stimuli.

Flags added for false friends:
  ff_english_word  the word is also an English word (WordNet lemma or Zipf >= 3 in English); the
                   original selection rule excluded these, so analysis leaves them out
  ff_other_class   the reviewer noted that the word means another ImageNet class (sensitivity check)

Inputs: stimuli/words_prereview.csv and stimuli/manifest_prereview.csv (written by build_stimuli.py)
and the reviewer's decisions in stimuli/review/spanish_translations_review.csv and
stimuli/review/spanish_false_friends_review.csv.
Outputs: stimuli/words.csv, stimuli/manifest.csv, stimuli/review_changes.csv.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nltk
import pandas as pd
from wordfreq import zipf_frequency

from common import ROOT, ortho_sim

nltk.data.path.insert(0, os.path.join(ROOT, "data", "nltk_data"))
from nltk.corpus import wordnet as wn  # noqa: E402

REVIEW = {"Translations": os.path.join(ROOT, "stimuli", "review", "spanish_translations_review.csv"),
          "False friends": os.path.join(ROOT, "stimuli", "review", "spanish_false_friends_review.csv")}
words = pd.read_csv(os.path.join(ROOT, "stimuli", "words_prereview.csv"))
man = pd.read_csv(os.path.join(ROOT, "stimuli", "manifest_prereview.csv"))
english_lexicon = {l.replace("_", " ").lower() for l in wn.all_lemma_names()}


def is_english_word(w):
    w = w.lower().strip()
    return w in english_lexicon or zipf_frequency(w, "en") >= 3.0


def norm(s):
    return str(s).strip()


def yes(v):
    return norm(v).upper().startswith("Y")


words["source"] = "omw"
words.loc[words.word_type.isin(["pseudoword", "anagram", "random"]), "source"] = "generated"
words.loc[words.word_type == "english", "source"] = "imagenet"
words["review_status"] = "not_reviewed"
words.loc[words.word_type.isin(["english", "pseudoword", "anagram", "random"]), "review_status"] = "control"
words["review_note"] = ""
words["ff_other_class"] = False

sheets = {
    "Translations": ("translation", ["en", "es", "sim", "hint", "ok", "better", "notes"]),
    "False friends": ("false_friend", ["en", "es", "sim", "hint", "ok", "better", "notes"]),
}
new_rows, changes, unmatched = [], [], []
for sheet, (wtype, cols) in sheets.items():
    df = pd.read_csv(REVIEW[sheet])
    df.columns = cols
    for _, r in df.iterrows():
        mask = (words.word_type == wtype) & (words.lang == "es") & (words.target_en == norm(r.en)) & (words.word == norm(r.es))
        if mask.sum() != 1:
            unmatched.append((sheet, r.en, r.es, int(mask.sum())))
            continue
        i = words.index[mask][0]
        note = "" if pd.isna(r.notes) else norm(r.notes)
        words.at[i, "review_note"] = note
        if wtype == "false_friend" and re.search(r"class|overlaps", note):
            words.at[i, "ff_other_class"] = True
        if yes(r.ok):
            words.at[i, "review_status"] = "accepted"
            continue
        words.at[i, "review_status"] = "rejected"
        better = "" if pd.isna(r.better) else norm(r.better).lower()
        changes.append({"sheet": sheet, "target_en": r.en, "old_word": r.es, "new_word": better or "(dropped)", "note": note})
        if not better:
            continue
        base = words.loc[i]
        new_rows.append({
            "target_idx": base.target_idx, "target_wnid": base.target_wnid, "target_en": base.target_en,
            "word_type": wtype, "lang": "es", "word": better, "substitutions": "",
            "sim_raw": round(ortho_sim(better, base.target_en, accents=True), 4),
            "sim_noaccent": round(ortho_sim(better, base.target_en, accents=False), 4),
            "n_chars": len(better), "zipf_en": zipf_frequency(better, "en"), "zipf_lang": zipf_frequency(better, "es"),
            "source": "reviewer", "review_status": "replacement", "review_note": f"replaces '{r.es}'",
            "ff_other_class": False,
        })

if unmatched:
    raise SystemExit(f"rows not matched to words.csv: {unmatched[:10]} (total {len(unmatched)})")

new = pd.DataFrame(new_rows)
keys = ["target_idx", "word_type", "lang", "word"]
clash = new.merge(words[keys], on=keys, how="inner")
if len(clash):
    raise SystemExit(f"replacement duplicates an existing row: {clash[keys].values.tolist()}")
words = pd.concat([words, new], ignore_index=True)
words["ff_english_word"] = [wt == "false_friend" and is_english_word(w) for wt, w in zip(words.word_type, words.word)]

# New stimuli: every replacement word on its target's existing source photos (same ranks).
photos = man[["target_idx", "source_idx", "source_wnid", "source_rank"]].drop_duplicates().sort_values(["target_idx", "source_rank"])
add = new[["target_idx", "target_en", "word_type", "lang", "word"]].merge(photos, on="target_idx")
add.insert(0, "stim_id", range(int(man.stim_id.max()) + 1, int(man.stim_id.max()) + 1 + len(add)))
man_out = pd.concat([man, add[man.columns]], ignore_index=True)

words.to_csv(os.path.join(ROOT, "stimuli", "words.csv"), index=False)
man_out.to_csv(os.path.join(ROOT, "stimuli", "manifest.csv"), index=False)
pd.DataFrame(changes).to_csv(os.path.join(ROOT, "stimuli", "review_changes.csv"), index=False)

es = words[words.lang == "es"]
print("Spanish review status by word type:")
print(es.groupby(["word_type", "review_status"]).size().to_string())
print(f"\nnew word rows: {len(new)}   new stimuli: {len(add)}   manifest rows now: {len(man_out):,}")
print("photos per new word:", add.groupby(["target_idx", "word_type", "word"]).size().value_counts().to_dict())
flag_en = words[words.ff_english_word]
print("\nfalse friends that are also English words (left out of analysis):", flag_en[["target_en", "word", "source"]].values.tolist())
print("false friends flagged as naming another class:", words[words.ff_other_class][["target_en", "word"]].values.tolist())
