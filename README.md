# Cross-Lingual Typographic Attacks on Contrastive Vision-Language Models

Companion repository for the study *Spelling Over Meaning: Cross-Lingual Typographic Attacks on
Contrastive Vision-Language Models* by Maikel Leon (University of Miami).

The study writes words on ImageNet photographs and records how strongly each word pulls the
zero-shot predictions of six CLIP models toward the ImageNet class it names. The written words
are English class names, their translations into Spanish, French, Portuguese, Italian, and
Catalan, false friends in Spanish and French, pseudowords that keep a graded share of the
letters of the English name, anagrams, and random consonant strings. Translations carry meaning
with varying spelling overlap, false friends carry spelling without meaning, and pseudowords carry
spelling alone, which separates what a model reads from the letters from what it reads from the
meaning.

This repository holds the word lists, the image manifest, the per-stimulus outputs of all six
models, and the code that builds the stimuli, runs the models, and produces every number, table,
and figure reported in the manuscript.

## Contents

```
code/                      stimulus construction, model runs, and analysis
  common.py                shared settings: models, rendering, spelling similarity
  build_stimuli.py         builds the pre-review word list and image manifest
  apply_review.py          applies the native-speaker review of the Spanish words
  apply_secondary_check.py applies the check of the other four languages and French false friends
  run_models.py            runs every stimulus through the six models (resumable)
  analyze.py, synthesis.py regression analyses used in the manuscript
  compare_review.py        compares key estimates before and after the lexical review
scripts/
  fetch_data.py            downloads and verifies the photographs and NLTK corpora
  fetch_models.py          downloads and verifies the six model checkpoints
  verify_sample.py         recomputes a sample of stimuli to check a local setup
  compute_facts.py         computes the reported numbers, the plot data, and the tables of models and words
  robustness.py            runs the robustness checks and writes the table of effects
  make_figures.py          draws the five manuscript figures
stimuli/
  words.csv                final word list (one row per written word)
  manifest.csv             final image manifest (one row per stimulus image)
  words_prereview.csv      word list before the lexical review
  manifest_prereview.csv   image manifest before the lexical review
  review_changes.csv       every change made in the Spanish review
  review/                  the Spanish reviewer's decisions, one file per word type
results/                   per-stimulus outputs and clean-photo log-probabilities per model
outputs/                   numbers (facts.json, robustness.json), plot data, LaTeX tables, and figures
photos/                    the public-domain photograph shown in Figure 2, with a record of its source
fonts/                     DejaVu Sans 2.37, the typeface used to render every word
checksums/                 SHA-256 sums of the photographs, corpora, font, and checkpoints
```

## Requirements

Python 3.13 and the packages pinned in `requirements.txt`:

```
pip install -r requirements.txt
```

PyTorch can be installed from the CPU wheel index if no GPU is available
(`pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu`).
All model runs in the study used the CPU, which keeps the outputs identical across machines up to
the four decimals stored in the result files.

## Reproducing the tables and figures

The shipped results are enough to regenerate every reported number, table, and figure, in about a
minute on a laptop:

```
python3 scripts/compute_facts.py    # outputs/data/facts.json, plot data, table_models.tex, table_stimuli.tex
python3 scripts/robustness.py       # outputs/data/robustness.json and table_effects.tex
python3 scripts/make_figures.py     # outputs/figures/*.pdf and *.png
```

Figure 2 writes six of the words on a public-domain photograph that is not part of the stimulus set
and is included in `photos/`, so it needs neither the ImageNet photographs nor the checkpoints. The
model predictions printed under that figure are cached in `outputs/data/fig2_predictions.json`;
pass `--recompute` to `make_figures.py` to rerun the two models it shows, which needs the
checkpoints described below.

## Checking a local setup

```
python3 scripts/fetch_data.py               # photographs and NLTK corpora, verified by checksum
python3 scripts/fetch_models.py             # model checkpoints, verified by checksum
python3 scripts/verify_sample.py rn50_yfcc15m 64
```

`verify_sample.py` rerenders a random sample of stimuli, runs them through the chosen model, and
compares the predictions and log-probabilities with the shipped result file.

## Full reproduction

1. Download and verify the inputs with `fetch_data.py` and `fetch_models.py`.
2. Optionally rebuild the stimulus set:
   ```
   python3 code/build_stimuli.py           # stimuli/words_prereview.csv, manifest_prereview.csv
   python3 code/apply_review.py            # stimuli/words.csv, manifest.csv, review_changes.csv
   python3 code/apply_secondary_check.py   # final stimuli/words.csv and manifest.csv
   ```
   Starting from the shipped pre-review files, the two review scripts reproduce the final
   `words.csv` and `manifest.csv` exactly. Rebuilding the pre-review list itself reproduces every
   English name, translation, pseudoword, anagram, and random string, but about 4% of the false
   friends differ: in the original build, ties between equally good false-friend candidates were
   broken in an order that depended on Python's per-process string hashing. The selection now sorts
   its candidates, so rebuilds are deterministic, and the shipped pre-review list remains the one
   used in the study.
3. Run the models: `python3 code/run_models.py`. Runs are resumable and append to
   `results/<model>_stimuli.csv`; on a laptop CPU the full set takes several hours per model.
   Set `TORCH_THREADS` to use more cores.
4. Regenerate the analysis with `compute_facts.py`, `robustness.py`, and `make_figures.py`, in that
   order.

## Models

| Key | Image encoder | Training data | Checkpoint source |
|---|---|---|---|
| `vitb32_laion400m` | ViT-B/32 | LAION-400M | OpenCLIP release v0.2 |
| `vitb16_laion400m` | ViT-B/16 | LAION-400M | OpenCLIP release v0.2 |
| `rn50_openai` | ResNet-50 | OpenAI WIT | OpenAI model server |
| `rn50_yfcc15m` | ResNet-50 | YFCC15M | OpenCLIP release v0.2 |
| `rn101_yfcc15m` | ResNet-101 | YFCC15M | OpenCLIP release v0.2 |
| `rn50_cc12m` | ResNet-50 | CC12M | OpenCLIP release v0.2 |

Classification is zero-shot over the 1,000 ImageNet classes with the prompt "a photo of a {class}."
and the class names distributed with OpenCLIP.

## Data dictionary

`stimuli/words.csv` (one row per written word)

| Column | Meaning |
|---|---|
| `target_idx`, `target_wnid`, `target_en` | ImageNet index, WordNet ID, and English name of the target class |
| `word_type` | `english`, `translation`, `false_friend`, `pseudoword`, `anagram`, or `random` |
| `lang` | `en`, `es`, `fr`, `pt`, `it`, `ca`, or `none` for strings without a language |
| `word` | the written word |
| `substitutions` | number of letters replaced (pseudowords only) |
| `sim_raw`, `sim_noaccent` | normalized Levenshtein similarity to the English name, with and without diacritics |
| `n_chars` | length of the word |
| `zipf_en`, `zipf_lang` | Zipf frequency in English and in the word's language |
| `source` | `imagenet`, `omw` (Open Multilingual Wordnet), `generated`, `reviewer` (replacement proposed in the Spanish review), or `criteria_check` (replacement from the second check) |
| `review_status` | `control`, `accepted`, `rejected` (excluded from analysis), or `replacement` |
| `review_note` | note recorded during the review |
| `ff_other_class` | false friend whose meaning names another ImageNet class (sensitivity analysis) |
| `ff_english_word` | false friend that becomes an English word once diacritics are removed (excluded from analysis) |
| `review_by` | `spanish_reviewer` or `criteria_check` for reviewed words, empty for controls |

`stimuli/manifest.csv` (one row per stimulus image): `stim_id`, the target and word columns above,
`source_idx` and `source_wnid` (the ImageNet class of the photograph the word is written on), and
`source_rank` (0 to 15; the ViT-B/32 model uses all 16 photographs per target and the other models
the first 10).

`results/<model>_stimuli.csv` (one row per stimulus): `pred` (top-1 class index), `target_logp`
(log-probability of the target class), `target_rank` (number of classes ranked above the target),
`true_logp` (log-probability of the photographed class), and `d_target_logp` and `d_true_logp`
(the same quantities minus their values on the clean photograph).

`results/<model>_clean.npz` holds `logp`, a 1,000 by 1,000 array of log-probabilities for the
clean photographs (rows) over the classes (columns). `results/cross_model_summary.csv` holds the
per-model estimates, and `results/cross_model_summary_prereview.csv` the same estimates computed
with the word list before the lexical review.

## Lexical review

A native speaker of Spanish reviewed all Spanish translations and false friends; the decisions are
in `stimuli/review/` and the resulting changes in `stimuli/review_changes.csv`. Translations in
French, Portuguese, Italian, and Catalan and the French false friends were checked against the same
criteria (`review_by = criteria_check`); those corrections are listed in
`code/apply_secondary_check.py`. Rejected words stay in the files and are excluded from analysis,
and replacement words were written on the same photographs as the words they replaced.

## Third-party materials

The ImageNet photographs and the model checkpoints are not redistributed here. `fetch_data.py`
downloads the photographs from the public sample at
https://github.com/EliSchwartz/imagenet-sample-images (commit `bc69043`), which remain subject to
the ImageNet terms of access, and the WordNet and Open Multilingual Wordnet corpora from the NLTK
data repository, under their own licenses. `fetch_models.py` downloads the checkpoints from the
OpenCLIP and OpenAI release locations, under the licenses of those projects. The DejaVu Sans font
is included under its license in `fonts/LICENSE-DejaVu.txt`. The photograph in `photos/`, shown in
Figure 2, was taken by an employee of the Federal Bureau of Investigation as part of official
duties and is in the public domain in the United States; `photos/README.md` records its source.

## License

Code is released under the MIT License (`LICENSE`). The word lists, review records, manifests,
model outputs, and derived tables are released under the Creative Commons Attribution 4.0
International License (`DATA_LICENSE.md`).

## Citation

Please cite the manuscript when it becomes available; until then, cite this repository using
`CITATION.cff`.
