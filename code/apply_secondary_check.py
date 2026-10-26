"""Apply the second lexical check of the French, Portuguese, Italian and Catalan translations and of
the French false friends, which used the same rules as the Spanish review: the everyday word for the class,
no wrong senses, genus or family names, plural forms, literal glosses, clippings or misspellings;
false friends must be unrelated in meaning to the class, not vulgar or offensive, and real words.

Corrections are applied like the Spanish review: rejected rows are kept but excluded from analysis,
replacements become new rows with new stimuli on the same photos. Rows not corrected are marked
accepted. Every row checked here gets review_by='criteria_check' (the Spanish rows get
'spanish_reviewer').
Run after apply_review.py; it refuses to run twice.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nltk
import pandas as pd
from wordfreq import zipf_frequency

from common import ROOT, ortho_sim, strip_accents

nltk.data.path.insert(0, os.path.join(ROOT, "data", "nltk_data"))
from nltk.corpus import wordnet as wn  # noqa: E402

ENGLISH = {l.replace("_", " ").lower() for l in wn.all_lemma_names()}


def is_english_word(w):
    """Accent-blind: a rendered 'émotion' reads as the English word 'emotion'."""
    w = w.lower().strip()
    s = strip_accents(w)
    return any(x in ENGLISH or zipf_frequency(x, "en") >= 3.0 for x in (w, s))

# (English class, language) -> replacement translation, or None to drop without replacement.
TRANSLATIONS = {
    ("accordion", "pt"): "acordeão", ("airliner", "pt"): "avião", ("airliner", "ca"): "avió",
    ("ant", "fr"): "fourmi", ("apiary", "ca"): "abellar", ("baboon", "ca"): "babuí",
    ("backpack", "fr"): "sac à dos", ("bakery", "it"): "panetteria", ("bakery", "ca"): "fleca",
    ("balloon", "it"): "palloncino", ("barbell", "fr"): "barre de musculation",
    ("barbell", "pt"): "barra de musculação", ("barbell", "ca"): "barra de pesos", ("barrel", "ca"): "bóta",
    ("baseball", "it"): "baseball", ("bassoon", "pt"): "fagote", ("bathtub", "fr"): "baignoire",
    ("bathtub", "ca"): "banyera", ("beach", "pt"): "praia", ("beach", "it"): "spiaggia",
    ("binoculars", "fr"): "jumelles", ("birdhouse", "it"): "casetta per uccelli",
    ("birdhouse", "ca"): "caseta per a ocells", ("bloodhound", "fr"): "chien de Saint-Hubert",
    ("bloodhound", "ca"): "gos de Sant Hubert", ("bolete", "ca"): "boletus", ("bookcase", "ca"): "prestatgeria",
    ("bra", "pt"): "sutiã", ("breakwater", "it"): "frangiflutti", ("breakwater", "ca"): "trencaones",
    ("breastplate", "fr"): "plastron", ("breastplate", "ca"): "plastró", ("bridegroom", "pt"): "noivo",
    ("broccoli", "fr"): "brocoli", ("bulbul", "fr"): "bulbul", ("bullmastiff", "ca"): "bullmastiff",
    ("bustard", "fr"): "outarde", ("cabbage", "ca"): "col", ("canoe", "fr"): "canoë", ("canoe", "ca"): "canoa",
    ("cassette", "fr"): "cassette", ("cassette", "pt"): "cassete", ("castle", "fr"): "château",
    ("cauldron", "fr"): "chaudron", ("cauldron", "it"): "calderone", ("cauldron", "ca"): "calder",
    ("centipede", "fr"): "mille-pattes", ("chain", "pt"): "corrente", ("chiffonier", "ca"): "calaixera",
    ("chiton", "ca"): "quitó", ("church", "pt"): "igreja", ("cleaver", "fr"): "couperet",
    ("cliff", "pt"): "penhasco", ("cliff", "it"): "rupe", ("clogs", "fr"): "sabots", ("cockroach", "ca"): "panerola",
    ("corn", "it"): "mais", ("cornet", "fr"): "trompette", ("cornet", "it"): "tromba", ("coucal", "fr"): "coucal",
    ("cougar", "fr"): "puma", ("cougar", "ca"): "puma", ("crayfish", "pt"): "lagostim",
    ("crossword", "pt"): "palavras cruzadas", ("cuirass", "ca"): "cuirassa", ("daisy", "pt"): "margarida",
    ("dam", "pt"): "barragem", ("dam", "ca"): "presa", ("desk", "it"): "scrivania", ("dishcloth", "ca"): "drap de cuina",
    ("dishwasher", "pt"): "lava-louça", ("dock", "fr"): "quai", ("dock", "it"): "molo", ("dough", "fr"): "pâte",
    ("dragonfly", "ca"): "libèl·lula", ("duck", "fr"): "canard", ("duck", "it"): "anatra", ("duck", "ca"): "ànec",
    ("echidna", "fr"): "échidné", ("eggnog", "pt"): "gemada", ("envelope", "pt"): "envelope",
    ("espresso", "ca"): "exprés", ("fireboat", "ca"): "vaixell de bombers", ("flatworm", "fr"): "ver plat",
    ("flatworm", "pt"): "verme plano", ("flatworm", "ca"): "cuc pla", ("fly", "fr"): "mouche",
    ("forklift", "ca"): "carretó elevador", ("fountain", "ca"): "font", ("gibbon", "fr"): "gibbon",
    ("goblet", "fr"): "verre à pied", ("goldfinch", "pt"): "pintassilgo", ("goldfish", "fr"): "poisson rouge",
    ("grasshopper", "fr"): "sauterelle", ("grasshopper", "pt"): "gafanhoto", ("greenhouse", "ca"): "hivernacle",
    ("guenon", "fr"): "cercopithèque", ("guenon", "ca"): "cercopitec", ("gyromitra", "ca"): None,
    ("harmonica", "fr"): "harmonica", ("harmonica", "ca"): "harmònica", ("harvestman", "fr"): "faucheux",
    ("harvestman", "pt"): "opilião", ("harvestman", "ca"): "opilió", ("hay", "fr"): "foin", ("hay", "pt"): "feno",
    ("hen", "fr"): "poule", ("hen", "ca"): "gallina", ("honeycomb", "ca"): "bresca",
    ("hourglass", "ca"): "rellotge de sorra", ("husky", "fr"): "husky", ("jacamar", "fr"): "jacamar",
    ("jacamar", "it"): "jacamar", ("jackfruit", "fr"): "jacque", ("jackfruit", "ca"): None, ("jay", "pt"): "gaio",
    ("lampshade", "ca"): "pantalla de llum", ("leopard", "pt"): "leopardo", ("lighter", "fr"): "briquet",
    ("lighter", "it"): "accendino", ("lighthouse", "fr"): "phare", ("limousine", "fr"): "limousine",
    ("limpkin", "fr"): "courlan", ("llama", "pt"): "lhama", ("lotion", "fr"): "lotion", ("lynx", "ca"): "linx",
    ("macaw", "ca"): "guacamai", ("magpie", "it"): "gazza", ("mailbox", "pt"): "caixa de correio",
    ("maraca", "ca"): "maraca", ("marimba", "fr"): "marimba", ("marimba", "it"): "marimba", ("marimba", "ca"): "marimba",
    ("maypole", "fr"): "arbre de mai", ("maypole", "ca"): "arbre de maig", ("meatloaf", "fr"): "pain de viande",
    ("meerkat", "ca"): "suricata", ("menu", "fr"): "menu", ("microphone", "fr"): "microphone",
    ("microphone", "ca"): "micròfon", ("miniskirt", "ca"): "minifaldilla", ("minivan", "ca"): "monovolum",
    ("mitten", "fr"): "moufle", ("monastery", "fr"): "monastère", ("mongoose", "pt"): "mangusto", ("mop", "it"): "mocio",
    ("mountain", "pt"): "montanha", ("mountain", "it"): "montagna", ("mushroom", "fr"): "champignon",
    ("mushroom", "pt"): "cogumelo", ("nematode", "fr"): "nématode", ("nematode", "it"): "nematode",
    ("orange", "it"): "arancia", ("oscilloscope", "fr"): "oscilloscope", ("paddle", "ca"): "pagaia",
    ("padlock", "pt"): "cadeado", ("paintbrush", "ca"): "pinzell", ("pedestal", "pt"): "pedestal",
    ("perfume", "fr"): "parfum", ("perfume", "ca"): "perfum", ("pier", "ca"): "embarcador", ("pig", "ca"): "porc",
    ("pillow", "pt"): "travesseiro", ("pineapple", "fr"): "ananas", ("plate", "ca"): "plat",
    ("plectrum", "fr"): "médiator", ("plectrum", "it"): "plettro", ("pomegranate", "pt"): "romã",
    ("pomegranate", "it"): "melagrana", ("popsicle", "fr"): "glace à l'eau", ("porcupine", "ca"): "porc espí",
    ("printer", "fr"): "imprimante", ("promontory", "ca"): "promontori", ("pug", "pt"): "pug",
    ("purse", "fr"): "sac à main", ("purse", "it"): "borsetta", ("purse", "ca"): "bossa de mà",
    ("quill", "fr"): "plume d'oie", ("quill", "it"): "penna d'oca", ("quilt", "pt"): "colcha",
    ("rooster", "ca"): "gall", ("safe", "fr"): "coffre-fort", ("sailboat", "fr"): "voilier", ("sailboat", "ca"): "veler",
    ("sandal", "pt"): "sandália", ("sandbar", "fr"): "banc de sable", ("sarong", "fr"): "paréo",
    ("saxophone", "it"): "sassofono", ("saxophone", "ca"): "saxòfon", ("scarf", "fr"): "écharpe",
    ("scarf", "pt"): "cachecol", ("scarf", "it"): "sciarpa", ("scarf", "ca"): "bufanda",
    ("scoreboard", "fr"): "tableau d'affichage", ("skunk", "fr"): "mouffette", ("skunk", "pt"): "cangambá",
    ("skunk", "it"): "moffetta", ("skunk", "ca"): "mofeta", ("snorkel", "fr"): "tuba", ("snorkel", "pt"): "snorkel",
    ("sock", "it"): "calzino", ("sock", "ca"): "mitjó", ("sombrero", "ca"): "barret mexicà", ("stage", "pt"): "palco",
    ("stage", "it"): "palcoscenico", ("stage", "ca"): "escenari", ("starfish", "fr"): "étoile de mer",
    ("starfish", "it"): "stella marina", ("stingray", "it"): "razza", ("stove", "fr"): "cuisinière",
    ("stove", "it"): "fornello", ("stove", "ca"): "fogons", ("strainer", "fr"): "passoire", ("strainer", "pt"): "coador",
    ("suit", "ca"): "vestit", ("sundial", "fr"): "cadran solaire", ("sunscreen", "pt"): "protetor solar",
    ("tank", "fr"): "char d'assaut", ("tank", "it"): "carro armato", ("thimble", "it"): "ditale",
    ("torch", "pt"): "tocha", ("trifle", "ca"): "trifle", ("trilobite", "fr"): "trilobite", ("tripod", "it"): "treppiede",
    ("trolleybus", "fr"): "trolleybus", ("umbrella", "fr"): "parapluie", ("umbrella", "it"): "ombrello",
    ("vase", "pt"): "vaso", ("vase", "it"): "vaso", ("vespa", "pt"): "scooter", ("vestment", "fr"): "chasuble",
    ("wallet", "fr"): "portefeuille", ("wardrobe", "pt"): "guarda-roupa", ("website", "fr"): "site web",
    ("weevil", "fr"): "charançon", ("weevil", "it"): "curculione", ("wheelbarrow", "ca"): "carretó",
    ("whistle", "it"): "fischietto",
}

# English class -> replacement French false friend, or None to drop.
FALSE_FRIENDS_FR = {
    "chiton": "coton", "violin": "vilain", "bookcase": "blocage", "joystick": "obstiné", "mop": "mois",
    "goose": "gousse", "accordion": "accroc", "basketball": None, "boxer": "bouger", "cardoon": "gardien",
    "consomme": "conformer", "cuirass": "cirage", "envelope": "développer", "harmonica": "pharmacien",
    "lakeshore": "accessoire", "lampshade": "ambassade", "library": "libre", "orange": "orage",
    "parachute": "arachide", "perfume": "perfide", "refrigerator": "regretter", "saxophone": "sarcophage",
    "volcano": "solvant", "plunger": "prunier", "projector": "protectorat",
}

# French false friends kept but whose meaning names another ImageNet class (sensitivity check).
FR_OTHER_CLASS = ["minibus", "coucal", "pole", "quail", "prison", "rotisserie", "baboon", "backpack", "beagle",
                  "buckle", "catamaran", "fig", "guillotine", "gyromitra", "komondor", "platypus", "stupa"]

words = pd.read_csv(os.path.join(ROOT, "stimuli", "words.csv"))
man = pd.read_csv(os.path.join(ROOT, "stimuli", "manifest.csv"))
if "review_by" in words.columns and (words["review_by"] == "criteria_check").any():
    raise SystemExit("already applied")
words["review_by"] = ""
words.loc[(words.lang == "es") & words.word_type.isin(["translation", "false_friend"]), "review_by"] = "spanish_reviewer"

scope = ((words.word_type == "translation") & words.lang.isin(["fr", "pt", "it", "ca"])) | \
        ((words.word_type == "false_friend") & (words.lang == "fr"))
words.loc[scope, "review_status"] = "accepted"
words.loc[scope, "review_by"] = "criteria_check"

new_rows, missing = [], []


def reject_and_replace(mask, new_word, wtype, lang, note):
    if mask.sum() != 1:
        missing.append((wtype, lang, note, int(mask.sum())))
        return
    i = words.index[mask][0]
    words.at[i, "review_status"] = "rejected"
    words.at[i, "review_note"] = note
    if new_word is None:
        return
    base = words.loc[i]
    new_rows.append({
        "target_idx": base.target_idx, "target_wnid": base.target_wnid, "target_en": base.target_en,
        "word_type": wtype, "lang": lang, "word": new_word, "substitutions": "",
        "sim_raw": round(ortho_sim(new_word, base.target_en, accents=True), 4),
        "sim_noaccent": round(ortho_sim(new_word, base.target_en, accents=False), 4),
        "n_chars": len(new_word), "zipf_en": zipf_frequency(new_word, "en"), "zipf_lang": zipf_frequency(new_word, lang),
        "source": "criteria_check", "review_status": "replacement", "review_note": f"replaces '{base.word}'",
        "ff_other_class": False, "ff_english_word": False, "review_by": "criteria_check",
    })


for (en, lang), new_word in TRANSLATIONS.items():
    m = (words.word_type == "translation") & (words.lang == lang) & (words.target_en == en) & (words.review_status != "replacement")
    reject_and_replace(m, new_word, "translation", lang, "criteria check")
for en, new_word in FALSE_FRIENDS_FR.items():
    m = (words.word_type == "false_friend") & (words.lang == "fr") & (words.target_en == en) & (words.review_status != "replacement")
    reject_and_replace(m, new_word, "false_friend", "fr", "criteria check")
words.loc[(words.word_type == "false_friend") & (words.lang == "fr") & words.target_en.isin(FR_OTHER_CLASS), "ff_other_class"] = True

if missing:
    raise SystemExit(f"rows not found: {missing}")
new = pd.DataFrame(new_rows)
keys = ["target_idx", "word_type", "lang", "word"]
clash = new.merge(words[words.review_status != "rejected"][keys], on=keys, how="inner")
if len(clash):
    raise SystemExit(f"replacement duplicates an existing row: {clash[keys].values.tolist()}")
words = pd.concat([words, new], ignore_index=True)
# Re-flag every false friend with the accent-blind English check (analysis leaves flagged ones out).
words["ff_english_word"] = [wt == "false_friend" and is_english_word(w) for wt, w in zip(words.word_type, words.word)]

photos = man[["target_idx", "source_idx", "source_wnid", "source_rank"]].drop_duplicates().sort_values(["target_idx", "source_rank"])
add = new[["target_idx", "target_en", "word_type", "lang", "word"]].merge(photos, on="target_idx")
add.insert(0, "stim_id", range(int(man.stim_id.max()) + 1, int(man.stim_id.max()) + 1 + len(add)))
words.to_csv(os.path.join(ROOT, "stimuli", "words.csv"), index=False)
pd.concat([man, add[man.columns]], ignore_index=True).to_csv(os.path.join(ROOT, "stimuli", "manifest.csv"), index=False)

chk = words[words.review_by == "criteria_check"]
print(chk.groupby(["word_type", "lang", "review_status"]).size().to_string())
ff = words[(words.word_type == "false_friend") & (words.review_status != "rejected")]
print(f"\nfalse friends in analysis: {(~ff.ff_english_word).sum()} of {len(ff)} "
      f"({ff.ff_english_word.sum()} read as English words once accents are ignored)")
print(f"\nnew word rows: {len(new)}   new stimuli: {len(add)}   manifest rows now: {len(man) + len(add):,}")
